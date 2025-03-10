#include <oneapi/dnnl/dnnl.hpp>
#include <level_zero/ze_api.h>
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

class sycl_args
{
public:
    sycl_args() = delete;
    sycl_args(sycl::buffer<uint8_t, 1, sycl::image_allocator, void> buf, bool isOutput) : _isBuf(true), _buf(buf), _val(0), _isOutput(isOutput)
    {
    }
    sycl_args(int val) : _isBuf(false), _buf(0, 1), _val(val)
    {
    }
    bool _isBuf;
    sycl::buffer<uint8_t, 1, sycl::image_allocator, void> _buf;
    int _val = 0; // if isBuf == false;
    bool _isOutput = false;
    friend std::ostream &operator<<(std::ostream &os, const sycl_args &bf);
};

std::ostream &operator<<(std::ostream &os, const sycl_args &bf)
{
    os << "sycl_args(_isBuf = " << bf._isBuf << ", _val = " << bf._val << ", _isOutput = " << bf._isOutput << ")";
    return os;
};

void my_set_args(sycl::handler &cgh, size_t idx, sycl_args buf) {
    std::cout << "** set args\n";
    if (buf._isOutput)
    {
        // Last one is output.
        sycl::accessor acc_param{buf._buf, cgh, sycl::read_write};
        cgh.set_arg(idx, acc_param);
    }
    else
    {
        if (buf._isBuf)
        {
            sycl::accessor acc_param{buf._buf, cgh, sycl::read_only};
            cgh.set_arg(idx, acc_param);
        }
        else
        {
            cgh.set_arg(idx, buf._val);
        }
    }
}

ze_module_handle_t myLoadModule(sycl::context ctxt, sycl::device device, const void *data, size_t dataSize)
{
    std::cout << "  == Start to load module(levelzero)" << std::endl;
    assert(data);
    ze_module_handle_t zeModule;
    ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             dataSize,
                             (const uint8_t *)data,
                             nullptr,
                             nullptr};
    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        device);
    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        ctxt);
    zeModuleCreate(zeContext, zeDevice, &desc, &zeModule, nullptr);
    return zeModule;
}

sycl::kernel *myGetKernel(sycl::context ctxt, ze_module_handle_t zeModule, const char *name)
{
    std::cout << "  == Start to make kernel based on zeModule." << std::endl;
    assert(zeModule);
    assert(name);
    ze_kernel_handle_t zeKernel;
    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;

    zeKernelCreate(zeModule, &desc, &zeKernel);
    sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {zeModule}, ctxt);

    auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {kernelBundle, zeKernel}, ctxt);

    // uint32_t groupSizeX = 32u;
    // uint32_t groupSizeY = 1u;
    // uint32_t groupSizeZ = 1u;
    // zeKernelSuggestGroupSize(zeKernel, 100, 1U, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
    // std::cout << "== suggest group: x=" << groupSizeX << ", y=" << groupSizeY << ", z=" << groupSizeZ << std::endl;
    // zeKernelSetGroupSize(zeKernel, groupSizeX, groupSizeY, groupSizeZ);

    return new sycl::kernel(kernel);
}

sycl::event myLaunchKernel(sycl::queue *queue, sycl::kernel *kernel, int element_size,
                           void **params, size_t paramsCount)
{
    std::cout << "  == Start to launch SPIRV kernel(add to sycl::queue)." << std::endl;
    return queue->submit([&](sycl::handler &cgh)
                         {
     for (size_t i = 0; i < paramsCount; i++) {
       cgh.set_arg(static_cast<uint32_t>(i), params[i]);
     }

    //  cgh.parallel_for(syclNdRange, *kernel); });
     cgh.parallel_for(sycl::range<1>(element_size), *kernel); });
}

sycl::event launchSPVKernelFromOpenCLOffline(sycl::queue &queue, size_t length, int32_t *X, int32_t *Y, int32_t *Z)
{
    std::cout << "== Start to launch SPIR-V kernel(converted from opencl kernel)." << std::endl;

    // Load SPIR-V binary
    std::string spirv_fn = "/mnt/users/odt/www/add.spv";
    std::ifstream spirv_file(spirv_fn, std::ios::binary);
    if (!spirv_file.is_open())
    {
        std::cout << "== Fail: Can't open file: " << spirv_fn << std::endl;
        exit(0);
    }
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());
    std::cout << "  == Readed spirv kernel file: " << spirv_fn << std::endl;

    // Create SYCL context and queue using Level Zero backend
    auto context = queue.get_context();
    auto device = queue.get_device();

    auto module = myLoadModule(context, device, spirv_binary.data(), spirv_binary.size());
    auto kernel = myGetKernel(context, module, "add_vectors");

    std::cout << "  == launch kernel" << std::endl;
    int32_t *params[3] = {X, Y, Z};
    return myLaunchKernel(&queue, kernel, length, reinterpret_cast<void **>(params), 3u);
}

void launchSPVKernelFromOpenCLOfflineLZ(ze_command_queue_handle_t queue, ze_context_handle_t context, ze_device_handle_t device, uint32_t length, int32_t *X, int32_t *Y, int32_t *Z)
{
    std::cout << "== Start to launch SPIR-V kernel(converted from opencl kernel)." << std::endl;

    // Load SPIR-V binary
    std::string spirv_fn = "/mnt/users/odt/www/add.spv";
    std::ifstream spirv_file(spirv_fn, std::ios::binary);
    if (!spirv_file.is_open())
    {
        std::cout << "== Fail: Can't open file: " << spirv_fn << std::endl;
        exit(0);
    }
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());
    std::cout << "  == Readed spirv kernel file: " << spirv_fn << std::endl;

    ze_module_handle_t zeModule;
    ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             spirv_binary.size(),
                             (const uint8_t *)spirv_binary.data(),
                             nullptr,
                             nullptr};
    zeModuleCreate(context, device, &desc, &zeModule, nullptr);
    ze_kernel_handle_t zeKernel;
    ze_kernel_desc_t desc_kernel = {};
    desc_kernel.pKernelName = "add_vectors";
    zeKernelCreate(zeModule, &desc_kernel, &zeKernel);


    zeKernelSetArgumentValue(zeKernel, 0, sizeof(X), &X);
    zeKernelSetArgumentValue(zeKernel, 1, sizeof(Y), &Y);
    zeKernelSetArgumentValue(zeKernel, 2, sizeof(Z), &Z);
    zeKernelSetArgumentValue(zeKernel, 3, sizeof(int), &length); // ‌:ml-citation{ref="4,6" data="citationList"}

    ze_group_count_t launchArgs = {length, 1, 1};
    zeKernelSetGroupSize(zeKernel, 64, 1, 1);

    ze_command_list_desc_t cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ze_command_list_handle_t cmdList;
    zeCommandListCreate(context, device, &cmdListDesc, &cmdList);

    zeCommandListAppendLaunchKernel(cmdList, zeKernel, &launchArgs, nullptr, 0, nullptr);
    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
    zeCommandQueueSynchronize(queue, UINT64_MAX); 







    // // Create SYCL context and queue using Level Zero backend
    // auto context = queue.get_context();
    // auto device = queue.get_device();

    // auto module = myLoadModule(context, device, spirv_binary.data(), spirv_binary.size());
    // auto kernel = myGetKernel(context, module, "add_vectors");

    // std::cout << "  == launch kernel" << std::endl;
    // int32_t *params[3] = {X, Y, Z};
    // return myLaunchKernel(&queue, kernel, length, reinterpret_cast<void **>(params), 3u);
}

sycl::event launchOpenCLKernelOnline(sycl::queue &q, size_t length, int32_t *X, int32_t *Z, int32_t offset, sycl::event &dep_event)
{
    std::cout << "== Start to test launch OpenCL kernel and compile online." << std::endl;
    // Kernel defined as an OpenCL C string.  This could be dynamically
    // generated instead of a literal.
    std::string source = R"""(
        __kernel void my_kernel(__global int *in, __global int *out, int offset_val) {
            size_t i = get_global_id(0);
            out[i] = out[i]*2 + in[i] + offset_val;
            printf("  == offset_val = %d, i = %d\n", offset_val, i);
        }
    )""";

    std::cout << "  == Start to kernel_bundle opencl source" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(
            q.get_context(),
            syclex::source_language::opencl,
            source);

    // Compile and link the kernel from the source definition.
    std::cout << "  == Start to kernel_bundle kb_src" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclex::build(kb_src);

    // Get a "kernel" object representing the kernel defined in the
    // source string.
    std::cout << "  == Start to get sycl::kernel" << std::endl;
    sycl::kernel k = kb_exe.ext_oneapi_get_kernel("my_kernel");

    // constexpr int N = length;
    constexpr int WGSIZE = 1;

#define UNIFY_DATA_TYPE 1
#if UNIFY_DATA_TYPE
    sycl::buffer inputbuf((uint8_t*)X, sycl::range{length*sizeof(int32_t)});
    sycl::buffer outputbuf((uint8_t*)Z, sycl::range{length*sizeof(int32_t)});

    std::vector<sycl_args> inputs_buf;
    inputs_buf.push_back(sycl_args(inputbuf, false));
    inputs_buf.push_back(sycl_args(outputbuf, true));
    inputs_buf.push_back(sycl_args(offset));
#else
    sycl::buffer inputbuf(X, sycl::range{length});
    sycl::buffer outputbuf(Z, sycl::range{length});
#endif
    std::cout << "  == Start to submit" << std::endl;

    // for (int i = 0; i < inputs_buf.size(); i++) {
    //     std::cout << "inputs[" << i << "] = " << inputs_buf[i] << std::endl;
    // }
    return q.submit([&](sycl::handler &cgh)
                    {
                        cgh.depends_on(dep_event);
#if UNIFY_DATA_TYPE
                        for (int i = 0; i < inputs_buf.size(); i++)
                        {
                            my_set_args(cgh, i, inputs_buf[i]);
                        }
#else
                        sycl::accessor in{inputbuf, cgh, sycl::read_only};
                        sycl::accessor out{outputbuf, cgh, sycl::read_write};
                        cgh.set_args(in, out); // All arguments
                        cgh.set_arg(2, offset); // scalar param
#endif
                        // Invoke the kernel over an nd-range.
                        sycl::nd_range ndr{{length}, {WGSIZE}};
                        cgh.parallel_for(ndr, k);
                    });
}

#define CHECK_ZE_RESULT(res, msg)                                               \
	if ((res) != ZE_RESULT_SUCCESS)                                             \
	{                                                                           \
		std::cerr << "Error: " << msg << " (Code: " << res << ")" << std::endl; \
		exit(EXIT_FAILURE);                                                     \
	}

auto init_level_zero()
{
	CHECK_ZE_RESULT(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit failed");

	uint32_t driver_count = 0;
	CHECK_ZE_RESULT(zeDriverGet(&driver_count, nullptr), "zeDriverGet count failed");
	std::vector<ze_driver_handle_t> drivers(driver_count);
	CHECK_ZE_RESULT(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet failed");
	ze_driver_handle_t driver = drivers[0];

	uint32_t device_count = 0;
	CHECK_ZE_RESULT(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet count failed");
	std::vector<ze_device_handle_t> devices(device_count);
	CHECK_ZE_RESULT(zeDeviceGet(driver, &device_count, devices.data()), "zeDeviceGet failed");
	ze_device_handle_t device = devices[0];

	ze_context_handle_t context;
	ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
	CHECK_ZE_RESULT(zeContextCreate(driver, &context_desc, &context), "zeContextCreate failed");

	return std::make_tuple(context, device);
}

auto get_sycl_info(ze_context_handle_t ze_context, ze_device_handle_t ze_device, ze_command_queue_handle_t ze_queue) {
	sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device> InteropDeviceInput{
		ze_device};

	sycl::device InteropDevice =
		sycl::make_device<sycl::backend::ext_oneapi_level_zero>(InteropDeviceInput);

	sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{
		ze_context, std::vector<sycl::device>(1, InteropDevice), sycl::ext::oneapi::level_zero::ownership::keep};

	sycl::context InteropContext =
		sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);

	sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> hQueueInteropInput = {ze_queue, InteropDevice};
	sycl::queue sycl_queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(hQueueInteropInput, InteropContext);

    return std::make_tuple(InteropDevice, InteropContext, sycl_queue);
}

auto init_oneDNN(sycl::device sycl_device, sycl::context sycl_context, sycl::queue sycl_queue) {
	if (sycl_queue.is_in_order()) {
		std::cout << "In-order queue" << std::endl;
	} else {
		std::cout << "Out-of-order queue" << std::endl;
	}
	auto eng = dnnl::sycl_interop::make_engine(sycl_device, sycl_context);
	auto strm = dnnl::sycl_interop::make_stream(eng, sycl_queue);

	return std::make_tuple(eng, strm);
}

int main()
{
	auto [context, device] = init_level_zero();

	ze_command_queue_handle_t command_queue;
	ze_command_queue_desc_t queue_desc = {};
	queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
	queue_desc.ordinal = 0;
	queue_desc.index = 0;
	queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS ;
	queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

	zeCommandQueueCreate(context, device, &queue_desc, &command_queue);

	ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
	ze_host_mem_alloc_desc_t host_mem_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
	int N = 102400;
	int32_t *A;
	int32_t *B;
	int32_t *C;
	int32_t *D;
	int32_t *E;
	int32_t *F;
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, N * sizeof(int32_t), 0, device, (void **)&A), "zeMemAllocShared failed for A");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, N * sizeof(int32_t), 0, device, (void **)&B), "zeMemAllocShared failed for B");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, N * sizeof(int32_t), 0, device, (void **)&C), "zeMemAllocShared failed for C");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, N * sizeof(int32_t), 0, device, (void **)&D), "zeMemAllocShared failed for D");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, N * sizeof(int32_t), 0, device, (void **)&E), "zeMemAllocShared failed for E");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, N * sizeof(int32_t), 0, device, (void **)&F), "zeMemAllocShared failed for E");

	for (size_t i = 0; i < N; ++i)
	{
		A[i] = static_cast<int32_t>(i);               // 0, 1, 2, 3, 4, 5, ...
		B[i] = static_cast<int32_t>(i * 2);           // 0, 2, 4, 6, 8, 10, ...
		C[i] = 0;
		D[i] = 0;
		E[i] = 0;
		F[i] = 0;
	}

    auto [sycl_device, sycl_context, sycl_queue] = get_sycl_info(context, device, command_queue);
	auto [oneDNN_eng, oneDNN_strm] = init_oneDNN(sycl_device, sycl_context, sycl_queue);

	dnnl::memory::dims shape = {N};
	auto mem_desc = dnnl::memory::desc(shape, dnnl::memory::data_type::s32, dnnl::memory::format_tag::x);
	dnnl::memory mem_A(mem_desc, oneDNN_eng, A);
	dnnl::memory mem_B(mem_desc, oneDNN_eng, B);
	dnnl::memory mem_C(mem_desc, oneDNN_eng, C);
	dnnl::memory mem_D(mem_desc, oneDNN_eng, D);
	dnnl::memory mem_E(mem_desc, oneDNN_eng, E);
	dnnl::memory mem_F(mem_desc, oneDNN_eng, F);

	auto mul_pd = dnnl::binary::primitive_desc(
		oneDNN_eng,
		dnnl::algorithm::binary_mul,
		mem_desc,
		mem_desc,
		mem_desc);
	dnnl::binary mul(mul_pd);

	auto add_pd = dnnl::binary::primitive_desc(
		oneDNN_eng,
		dnnl::algorithm::binary_add,
		mem_desc,
		mem_desc,
		mem_desc);
	dnnl::binary add(add_pd);

	sycl::event event = dnnl::sycl_interop::execute(
		mul, oneDNN_strm,
		{{DNNL_ARG_SRC_0, mem_A},
		 {DNNL_ARG_SRC_1, mem_B},
		 {DNNL_ARG_DST, mem_C}});

	ze_event_handle_t level0_event0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event);
	zeEventHostSynchronize(level0_event0, UINT64_MAX); // Wait for completion

	sycl::event event1 = dnnl::sycl_interop::execute(
		add, oneDNN_strm,
		{{DNNL_ARG_SRC_0, mem_C},
		 {DNNL_ARG_SRC_1, mem_A},
		 {DNNL_ARG_DST, mem_D}}, {event});

	ze_event_handle_t level0_event1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event);
	zeEventHostSynchronize(level0_event1, UINT64_MAX); // Wait for completion

	sycl::event event2 = dnnl::sycl_interop::execute(
		add, oneDNN_strm,
		{{DNNL_ARG_SRC_0, mem_A},
		 {DNNL_ARG_SRC_1, mem_B},
		 {DNNL_ARG_DST, mem_E}});

	ze_event_handle_t level0_event2 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event2);
	zeEventHostSynchronize(level0_event2, UINT64_MAX); // Wait for completion

	sycl::event event3 = dnnl::sycl_interop::execute(
		add, oneDNN_strm,
		{{DNNL_ARG_SRC_0, mem_D},
		 {DNNL_ARG_SRC_1, mem_E},
		 {DNNL_ARG_DST, mem_F}}, {event1, event2});

	ze_event_handle_t level0_event3 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event3);
	zeEventHostSynchronize(level0_event3, UINT64_MAX); // Wait for completion

    launchSPVKernelFromOpenCLOfflineLZ(command_queue, context, device, 10, A, F, F);

    launchOpenCLKernelOnline(sycl_queue, 10, A, F, 0, event3);


	for (size_t i = 0; i < 5; i++)
	{
		std::cout << "OUTPUT[" << i << "] = " << F[i] << std::endl;
	}
	return 0;
}