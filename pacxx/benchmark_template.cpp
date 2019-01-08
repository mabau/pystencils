#include <PACXX.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <chrono>


using namespace pacxx::v2;

size_t division_round_up(size_t a, size_t b)
{
    if( a % b == 0)
        return a / b;
    else
        return (a / b) + 1;
}

int main(int argc, char** argv)
{
    {% if target == 'cpu' %}
    Executor::Create<NativeRuntime>(0);
    {% elif target == 'gpu' %}
    Executor::Create<CUDARuntime>(0);
    {% endif %}

    if( argc != 5 ) {
        std::cout << "Usage:  ./benchmark xSize ySize zSize iterations" << std::endl;
        return 1;
    }
    Dimension3 domainSize;
    int64_t iterations;
    auto &exec = Executor::get(0);

    std::stringstream( argv[1] ) >> domainSize.x;
    std::stringstream( argv[2] ) >> domainSize.y;
    std::stringstream( argv[3] ) >> domainSize.z;
    std::stringstream( argv[4] ) >> iterations;

    // add ghost layers to be comparable to pystencils native backend
    domainSize.x += 2;
    domainSize.y += 2;
    domainSize.z += 2;

    int64_t totalSize = domainSize.x * domainSize.y * domainSize.z * {{f_size}};

    std::vector<double> src( totalSize, 0.0 );
    std::vector<double> dst( totalSize, 0.0 );

    auto & dsrc = exec.allocate<double>(src.size());
    auto & ddst = exec.allocate<double>(dst.size());

    dsrc.upload(src.data(), src.size());
    ddst.upload(dst.data(), dst.size());

    double * _data_src = dsrc.get();
    double * _data_dst = ddst.get();

    const int64_t _size_src_0 = domainSize.x;
    const int64_t _size_src_1 = domainSize.y;
    const int64_t _size_src_2 = domainSize.z;

    // fzyx layout
    const int64_t _stride_src_0 = 1;
    const int64_t _stride_src_1 = domainSize.x;
    const int64_t _stride_src_2 = domainSize.x * domainSize.y;
    const int64_t _stride_src_3 = domainSize.x * domainSize.y * domainSize.z;

    auto pacxxKernel = [=]( range & config ) {

        struct Vec3D {int x; int y; int z; };
        const Vec3D blockDim  = { config.get_block_size(0), config.get_block_size(1), config.get_block_size(2) };
        const Vec3D blockIdx  = { config.get_block(0), config.get_block(1), config.get_block(2) };
        const Vec3D threadIdx = { config.get_local(0), config.get_local(1), config.get_local(2) };

        {{ code|indent(8) }}
    };

    size_t blockSize[] = {64, 8, 1};

    KernelConfiguration config( { division_round_up(domainSize.x - 2, blockSize[0]),
                                  division_round_up(domainSize.y - 2, blockSize[1]),
                                  division_round_up(domainSize.z  -2, blockSize[2]) },
                                  { blockSize[0],
                                    blockSize[1],
                                    blockSize[2] });

    // warm up
    for( int64_t i = 0; i < 10; ++i ) {
        exec.launch(pacxxKernel, config);
    }
    exec.synchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for( int64_t i = 0; i < iterations; ++i ) {
        exec.launch(pacxxKernel, config);
    }
    exec.synchronize();
    auto duration = std::chrono::high_resolution_clock::now() - start;

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    std::cout << ns.count() * 1e-9 << std::endl;

}
