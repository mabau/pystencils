#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <array>
#include <string>
#include <sstream>

${includes}

namespace py = pybind11;

#define RESTRICT ${restrict_qualifier}

namespace internal {

${kernel_definition}

}

std::string tuple_to_str(const ssize_t * data, const size_t N){
    std::stringstream acc;
    acc << "(";
    for(size_t i = 0; i < N; ++i){
        acc << data[i];
        if(i + 1 < N){
            acc << ", ";
        }
    }
    acc << ")";
    return acc.str();
}

template< typename T >
void checkFieldShape(const std::string& fieldName, const std::string& expected, const py::array_t< T > & arr, size_t coord, size_t desired) {
    auto panic = [&](){
        std::stringstream err;
        err << "Invalid shape of argument " << fieldName
            << ". Expected " << expected
            << ", but got " << tuple_to_str(arr.shape(), arr.ndim())
            << ".";
        throw py::value_error{ err.str() };
    };
    
    if(arr.ndim() <= coord){
        panic();
    }

    if(arr.shape(coord) != desired){
        panic();
    }
}

template< typename T >
void checkFieldStride(const std::string fieldName, const std::string& expected, const py::array_t< T > & arr, size_t coord, size_t desired) {
    auto panic = [&](){
        std::stringstream err;
        err << "Invalid strides of argument " << fieldName 
            << ". Expected " << expected
            << ", but got " << tuple_to_str(arr.strides(), arr.ndim())
            << ".";
        throw py::value_error{ err.str() };
    };
    
    if(arr.ndim() <= coord){
        panic();
    }

    if(arr.strides(coord) / sizeof(T) != desired){
        panic();
    }
}

void check_params_${kernel_name} (${public_params}) {
${param_check_lines}
}

void run_${kernel_name} (${public_params}) {
${extraction_lines}
    internal::${kernel_name}(${kernel_args});
}

PYBIND11_MODULE(${module_name}, m) {
    m.def("check_params", &check_params_${kernel_name}, py::kw_only(), ${param_binds});
    m.def("invoke", &run_${kernel_name}, py::kw_only(), ${param_binds});
}
