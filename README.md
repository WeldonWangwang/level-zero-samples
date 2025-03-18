# level-zero-samples


### Build and run

source /opt/intel/oneapi/setvars.sh
cd level-zero-samples
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=icpx ..
make -j20
cd 00_onednn_with_l0
./00_onednn_with_l0 1
