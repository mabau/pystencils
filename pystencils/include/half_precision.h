/// Half precision support. Experimental. Use carefully.
///
/// This feature is experimental, since it strictly depends on the underlying architecture and compiler support.
/// On x86 architectures, what you can expect is that the data format is supported natively only for storage and
/// interchange. Arithmetic operations will likely involve casting to fp32 (C++ float) and truncation to fp16.
/// Only bandwidth bound code may therefore benefit. None of this is guaranteed, and may change in the future.

/// Clang version must be 15 or higher for x86 half precision support.
/// GCC version must be 12 or higher for x86 half precision support.
/// Also support seems to require SSE, so ensure that respective instruction sets are enabled.
/// See
///   https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point
///   https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html
/// for more information.

#pragma once
using half    = _Float16;
