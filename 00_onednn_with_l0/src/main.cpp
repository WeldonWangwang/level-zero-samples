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
#include <sstream>

namespace syclex = sycl::ext::oneapi::experimental;

#define CHECK_ZE_RESULT(res, msg)                                               \
	if ((res) != ZE_RESULT_SUCCESS)                                             \
	{                                                                           \
		std::cerr << "Error: " << msg << " (Code: " << res << ")" << std::endl; \
		exit(EXIT_FAILURE);                                                     \
	}

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

    return new sycl::kernel(kernel);
}

sycl::event myLaunchKernel(sycl::queue *queue, sycl::kernel *kernel, int element_size,
                           void **params, size_t paramsCount)
{
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
    // Load SPIR-V binary
    std::string spirv_fn = "/mnt/users/odt/www/add.spv";
    std::ifstream spirv_file(spirv_fn, std::ios::binary);
    if (!spirv_file.is_open())
    {
        std::cout << "== Fail: Can't open file: " << spirv_fn << std::endl;
        exit(0);
    }
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());

    // Create SYCL context and queue using Level Zero backend
    auto context = queue.get_context();
    auto device = queue.get_device();

    auto module = myLoadModule(context, device, spirv_binary.data(), spirv_binary.size());
    auto kernel = myGetKernel(context, module, "add_vectors");

    int32_t *params[3] = {X, Y, Z};
    return myLaunchKernel(&queue, kernel, length, reinterpret_cast<void **>(params), 3u);
}

ze_event_handle_t launchSPVKernelFromOpenCLOfflineLZ(ze_command_queue_handle_t queue, ze_context_handle_t context, ze_device_handle_t device, uint32_t length, float *X, float *Y, float *Z, ze_event_handle_t event)
{
    // Load SPIR-V binary
    std::string spirv_fn = "/mnt/users/odt/www/level-zero-samples/00_onednn_with_l0/matmul.spv";
    std::ifstream spirv_file(spirv_fn, std::ios::binary);
    if (!spirv_file.is_open())
    {
        std::cout << "== Fail: Can't open file: " << spirv_fn << std::endl;
        exit(0);
    }
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());

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
    desc_kernel.pKernelName = "matmul";
    zeKernelCreate(zeModule, &desc_kernel, &zeKernel);

    zeKernelSetArgumentValue(zeKernel, 0, sizeof(X), &X);
    zeKernelSetArgumentValue(zeKernel, 1, sizeof(Y), &Y);
    zeKernelSetArgumentValue(zeKernel, 2, sizeof(Z), &Z);
    zeKernelSetArgumentValue(zeKernel, 3, sizeof(int), &length);
    zeKernelSetArgumentValue(zeKernel, 4, sizeof(int), &length);
    zeKernelSetArgumentValue(zeKernel, 5, sizeof(int), &length);

    ze_group_count_t launchArgs = {length, length, length};
    zeKernelSetGroupSize(zeKernel, 64, 1, 1);

    ze_command_list_desc_t cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ze_command_list_handle_t cmdList;
    zeCommandListCreate(context, device, &cmdListDesc, &cmdList);

    ze_event_pool_desc_t eventPoolDesc = {};
    eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    eventPoolDesc.count = 1;
    ze_event_pool_handle_t eventPool;
    zeEventPoolCreate(context, &eventPoolDesc, 1, &device, &eventPool);

    ze_event_handle_t kernelEvent;
    ze_event_desc_t eventDesc = {};
    eventDesc.index = 0;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    zeEventCreate(eventPool, &eventDesc, &kernelEvent);
    zeCommandListAppendLaunchKernel(cmdList, zeKernel, &launchArgs, kernelEvent, 1, &event);
    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
    return kernelEvent;
}

sycl::event launchOpenCLKernelOnline(sycl::queue &q, sycl::kernel k, size_t length, float *X, float *Y, float *Z)
{
    sycl::buffer inputbuf((uint8_t*)X, sycl::range{length*length*sizeof(float)});
    sycl::buffer weightbuf((uint8_t*)Y, sycl::range{length*length*sizeof(float)});
    sycl::buffer outputbuf((uint8_t*)Z, sycl::range{length*length*sizeof(float)});

    std::vector<sycl_args> inputs_buf;
    inputs_buf.push_back(sycl_args(inputbuf, false));
    inputs_buf.push_back(sycl_args(weightbuf, false));
    inputs_buf.push_back(sycl_args(outputbuf, true));
    inputs_buf.push_back(sycl_args(length));
    inputs_buf.push_back(sycl_args(length));
    inputs_buf.push_back(sycl_args(length));

    // sycl::event sycl_event = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
    //     { ze_event, sycl::ext::oneapi::level_zero::ownership::keep }, q.get_context());
    return q.submit([&](sycl::handler &cgh)
                    {
                        // cgh.depends_on(sycl_event);
                        for (int i = 0; i < inputs_buf.size(); i++)
                        {
                            my_set_args(cgh, i, inputs_buf[i]);
                        }
                        // Invoke the kernel over an nd-range.
                        // sycl::nd_range ndr{{length*length}, {WGSIZE}};
                        // cgh.parallel_for(ndr, k);
                        cgh.parallel_for(sycl::range<2>(length, length), k);
                    });
}

sycl::kernel create_opencl_kernel_online(sycl::queue &q) {
    std::string source = R"""(
        __kernel void my_kernel(
            __global float* A,  // 输入矩阵 A
            __global float* B,  // 输入矩阵 B
            __global float* C,  // 输出矩阵 C
            const int M,        // A 的行数
            const int N,        // A 的列数 / B 的行数
            const int K         // B 的列数
        ) {
            // 获取当前 work-item 的全局 ID
            int row = get_global_id(0);  // 行索引
            int col = get_global_id(1);  // 列索引

            // 仅计算有效的 C[row, col]
            if (row < M && col < K) {
                float sum = 0.0f;

                // 计算点积 A[row, :] * B[:, col]
                for (int i = 0; i < N; i++) {
                    sum += A[row * N + i] * B[i * K + col];
                }

                // 存储到 C
                C[row * K + col] = sum;
            }
        }
    )""";

    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(
            q.get_context(),
            syclex::source_language::opencl,
            source);

    // Compile and link the kernel from the source definition.
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclex::build(kb_src);

    // Get a "kernel" object representing the kernel defined in the
    // source string.
    sycl::kernel k = kb_exe.ext_oneapi_get_kernel("my_kernel");
    return k;
}

sycl::event launch_sycl_kernel(sycl::queue sycl_queue, int M, int K, int N, float* src, float* weights, float* out, sycl::event dep){
    auto event5 = sycl_queue.submit([&](sycl::handler& h) {
            h.depends_on(dep);
            h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {
                int row = idx[0];
                int col = idx[1];

                float sum = 0.0f;
                for (int i = 0; i < K; i++) {
                    sum += src[row * K + i] * weights[i * N + col];
                }
                out[row * N + col] = sum / 1.0e+27f;
            });
        });
    return event5;
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

sycl::event onednn_mutamul_execute(dnnl::engine engine, dnnl::stream stream, dnnl::matmul matmul_prim, int M, int K, int N, float *src, float *weights, float *dst, const std::vector<sycl::event> &deps = {}) {
    dnnl::memory src_mem({{{M, K}}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, engine, src);
    dnnl::memory weights_mem({{{K, N}}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, engine, weights);
    dnnl::memory dst_mem({{{M, N}}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, engine, dst);

	sycl::event event = dnnl::sycl_interop::execute(
		matmul_prim, stream,
		{{DNNL_ARG_SRC, src_mem},
		 {DNNL_ARG_WEIGHTS, weights_mem},
		 {DNNL_ARG_DST, dst_mem}}, deps);
    return event;
}

dnnl::matmul create_onednn_kernel(dnnl::engine engine, int M, int K, int N) {
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, 
        dnnl::memory::desc({M, K}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab),
        dnnl::memory::desc({K, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab),
        dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab)
    );

    auto matmul_prim = dnnl::matmul(matmul_pd);
    
    return matmul_prim;
}

int main(int argc, char* argv[]) {
    bool enable_lz_event;
    if (argc < 2) {
        std::cout << "Please set if enable Level Zero event control" << std::endl;
        return 1;
    }
    std::string arg = argv[1];
    if (arg == "true" || arg == "1") {
        enable_lz_event = true;
    } else if (arg == "false" || arg == "0") {
        enable_lz_event = false;
    } else {
        return 1;
    }

	auto [context, device] = init_level_zero();

	ze_command_queue_handle_t command_queue;
	ze_command_queue_desc_t queue_desc = {};
	queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
	queue_desc.ordinal = 0;
	queue_desc.index = 0;
	queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS ;
	queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

	zeCommandQueueCreate(context, device, &queue_desc, &command_queue);

	ze_device_mem_alloc_desc_t device_mem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
	ze_host_mem_alloc_desc_t host_mem_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    auto [sycl_device, sycl_context, sycl_queue] = get_sycl_info(context, device, command_queue);
	auto [oneDNN_eng, oneDNN_strm] = init_oneDNN(sycl_device, sycl_context, sycl_queue);

	const int m = 5120;
	float *A;
	float *B;
	float *C;
	float *D;
	float *E;
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, m * m * sizeof(float), 64, device, (void **)&A), "zeMemAllocShared failed for A");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, m * m * sizeof(float), 64, device, (void **)&B), "zeMemAllocShared failed for B");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, m * m * sizeof(float), 64, device, (void **)&C), "zeMemAllocShared failed for C");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, m * m * sizeof(float), 64, device, (void **)&D), "zeMemAllocShared failed for C");
	CHECK_ZE_RESULT(zeMemAllocShared(context, &device_mem_desc, &host_mem_desc, m * m * sizeof(float), 64, device, (void **)&E), "zeMemAllocShared failed for C");


    for (int i = 0; i < m * m; i++) A[i] = i % 2 + 1;   // A: 1,2,1,...
    for (int i = 0; i < m * m; i++) B[i] = i % 2 + 1; // B: 1,2,1,...
    for (int i = 0; i < m * m; i++) C[i] = 0; // C = 0
    for (int i = 0; i < m * m; i++) D[i] = 0; // D = 0
    for (int i = 0; i < m * m; i++) E[i] = 0; // D = 0

    auto matmul_prim = create_onednn_kernel(oneDNN_eng, m, m, m);
    auto matmul_opencl = create_opencl_kernel_online(sycl_queue);

    sycl::event event = launchOpenCLKernelOnline(sycl_queue, matmul_opencl, m, A, B, C);
    // sycl event -> sycl
    sycl::event event1 = onednn_mutamul_execute(oneDNN_eng, oneDNN_strm, matmul_prim, m, m, m, C, B, D, {event});
    // sycl event -> sycl
    sycl::event event2 = onednn_mutamul_execute(oneDNN_eng, oneDNN_strm, matmul_prim, m, m, m, D, A, C, {event1});
    // sycl event -> L0
    ze_event_handle_t level0_event2 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event2);
    ze_event_handle_t level0_event3 = launchSPVKernelFromOpenCLOfflineLZ(command_queue, context, device, 512, C, D, E, level0_event2);
    // L0 event -> L0
    ze_event_handle_t level0_event4 = launchSPVKernelFromOpenCLOfflineLZ(command_queue, context, device, 512, A, E, C, level0_event3);
    // L0 event -> sycl
    sycl::event sycl_event4 = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        { level0_event4, sycl::ext::oneapi::level_zero::ownership::keep }, sycl_queue.get_context());
    sycl::event event5 = launch_sycl_kernel(sycl_queue, m, m, m, B, C, D, sycl_event4);

    if (enable_lz_event) {
        ze_event_handle_t level0_event5 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(event5);
        zeEventHostSynchronize(level0_event5, UINT64_MAX); // Wait for completion
    }
    const float expected[] = {1.2292};
    bool success = true;
    if (std::abs(D[0] - expected[0]) > 1e-4) {
        std::cout << D[0] << " " << expected[0] << std::endl;
        std::cout << std::abs(D[0] - expected[0]) << std::endl;
        success = false;
    }
    if (B[0] == expected[0]) {
        std::cout << "pass\n";
    }
    std::cout << "Result verification: " 
                << (success ? "PASS" : "FAIL") 
                << std::endl;
    zeMemFree(context, A);
    zeMemFree(context, B);
    zeMemFree(context, C);
    zeMemFree(context, D);
    zeMemFree(context, E);
    zeContextDestroy(context);

    return 0;
}