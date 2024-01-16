#pragma once

extern "C++" {
#ifdef __CUDA_ARCH__
template <typename DTYPE_T, std::size_t DIMENSION> struct PyStencilsField {
  DTYPE_T *data;
  DTYPE_T shape[DIMENSION];
  DTYPE_T stride[DIMENSION];
};
#else
#include <array>

template <typename DTYPE_T, std::size_t DIMENSION> struct PyStencilsField {
  DTYPE_T *data;
  std::array<DTYPE_T, DIMENSION> shape;
  std::array<DTYPE_T, DIMENSION> stride;
};
#endif
}
