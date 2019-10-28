# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import hashlib
import itertools
from enum import Enum
from typing import Set

import sympy as sp
from sympy.core.cache import cacheit

import pystencils
from pystencils.astnodes import Node
from pystencils.data_types import TypedSymbol, cast_func, create_type

try:
    import pycuda.driver
except Exception:
    pass

_hash = hashlib.md5


class InterpolationMode(str, Enum):
    NEAREST_NEIGHBOR = "nearest_neighbour"
    NN = NEAREST_NEIGHBOR
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"


class Interpolator(object):
    """
    Implements non-integer accesses on fields using linear interpolation.

    On GPU, this interpolator can be implemented by a :class:`.TextureCachedField` for hardware acceleration.

    Address modes are different boundary handlings possible choices are like for CUDA textures

        **CLAMP**

        The signal c[k] is continued outside k=0,...,M-1 so that c[k] = c[0] for k < 0, and c[k] = c[M-1] for k >= M.

        **BORDER**

        The signal c[k] is continued outside k=0,...,M-1 so that c[k] = 0 for k < 0and for k >= M.

        Now, to describe the last two address modes, we are forced to consider normalized coordinates,
        so that the 1D input signal samples are assumed to be c[k / M], with k=0,...,M-1.

        **WRAP**

        The signal c[k / M] is continued outside k=0,...,M-1 so that it is periodic with period equal to M.
        In other words, c[(k + p * M) / M] = c[k / M] for any (positive, negative or vanishing) integer p.

        **MIRROR**

        The signal c[k / M] is continued outside k=0,...,M-1 so that it is periodic with period equal to 2 * M - 2.
        In other words, c[l / M] = c[k / M] for any l and k such that (l + k)mod(2 * M - 2) = 0.

    Explanations from https://stackoverflow.com/questions/19020963/the-different-addressing-modes-of-cuda-textures
    """

    required_global_declarations = []

    def __init__(self,
                 parent_field,
                 interpolation_mode: InterpolationMode,
                 address_mode='BORDER',
                 use_normalized_coordinates=False,
                 allow_textures=True):
        super().__init__()

        self.field = parent_field.new_field_with_different_name(parent_field.name)
        self.field.field_type = pystencils.field.FieldType.CUSTOM
        self.address_mode = address_mode
        self.use_normalized_coordinates = use_normalized_coordinates
        hash_str = "%x" % abs(hash(self.field) + hash(address_mode))
        self.symbol = TypedSymbol('dummy_symbol_carrying_field' + self.field.name + hash_str,
                                  'dummy_symbol_carrying_field' + self.field.name + hash_str)
        self.symbol.field = self.field
        self.symbol.interpolator = self
        self.allow_textures = allow_textures
        self.interpolation_mode = interpolation_mode

    @property
    def _hashable_contents(self):
        return (str(self.address_mode),
                str(type(self)),
                self.symbol,
                self.address_mode,
                self.use_normalized_coordinates)

    def at(self, offset):
        return InterpolatorAccess(self.symbol, *[sp.S(o) for o in offset])

    def __getitem__(self, offset):
        return InterpolatorAccess(self.symbol, *[sp.S(o) for o in offset])

    def __str__(self):
        return '%s_interpolator_%s' % (self.field.name, self.reproducible_hash)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self._hashable_contents)

    @property
    def reproducible_hash(self):
        return _hash(str(self._hashable_contents).encode()).hexdigest()


class LinearInterpolator(Interpolator):

    def __init__(self,
                 parent_field: pystencils.Field,
                 address_mode='BORDER',
                 use_normalized_coordinates=False):
        super().__init__(parent_field,
                         InterpolationMode.LINEAR,
                         address_mode,
                         use_normalized_coordinates)


class NearestNeightborInterpolator(Interpolator):

    def __init__(self,
                 parent_field: pystencils.Field,
                 address_mode='BORDER',
                 use_normalized_coordinates=False):
        super().__init__(parent_field,
                         InterpolationMode.NN,
                         address_mode,
                         use_normalized_coordinates)


class InterpolatorAccess(TypedSymbol):
    def __new__(cls, field, offsets, *args, **kwargs):
        obj = TextureAccess.__xnew_cached_(cls, field, offsets, *args, **kwargs)
        return obj

    def __new_stage2__(self, symbol, *offsets):
        assert offsets is not None
        obj = super().__xnew__(self, '%s_interpolator_%x' %
                               (symbol.field.name, abs(hash(tuple(offsets)))), symbol.field.dtype)
        obj.offsets = offsets
        obj.symbol = symbol
        obj.field = symbol.field
        obj.interpolator = symbol.interpolator
        return obj

    def __hash__(self):
        return hash((self.symbol, self.field, tuple(self.offsets), self.interpolator))

    def __str__(self):
        return '%s_interpolator(%s)' % (self.field.name, ','.join(str(o) for o in self.offsets))

    def __repr__(self):
        return self.__str__()

    def atoms(self, *types):
        if self.offsets:
            offsets = set(o for o in self.offsets if isinstance(o, types))
            if isinstance(self, *types):
                offsets.update([self])
            for o in self.offsets:
                if hasattr(o, 'atoms'):
                    offsets.update(set(o.atoms(types)))
            return offsets
        else:
            return set()

    @property
    def free_symbols(self):
        symbols = set()
        if self.offsets is not None:
            for o in self.offsets:
                if hasattr(o, 'free_symbols'):
                    symbols.update(set(o.free_symbols))
                # if hasattr(o, 'atoms'):
                    # symbols.update(set(o.atoms(sp.Symbol)))

        return symbols

    @property
    def args(self):
        return [self.symbol, *self.offsets]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        return {self}

    @property
    def interpolation_mode(self):
        return self.interpolator.interpolation_mode

    def implementation_with_stencils(self):
        field = self.field

        default_int_type = create_type('int64')
        use_textures = isinstance(self, TextureAccess)
        if use_textures:
            def absolute_access(x, _):
                return self.texture.at((o for o in x))
        else:
            absolute_access = field.absolute_access

        sum = [0, ] * (field.shape[0] if field.index_dimensions else 1)

        offsets = self.offsets
        rounding_functions = (sp.floor, lambda x: sp.floor(x) + 1)

        for channel_idx in range(field.shape[0] if field.index_dimensions else 1):
            if self.interpolation_mode == InterpolationMode.NN:
                if use_textures:
                    sum[channel_idx] = self
                else:
                    sum[channel_idx] = absolute_access([sp.floor(i + 0.5) for i in offsets], channel_idx)

            elif self.interpolation_mode == InterpolationMode.LINEAR:
                # TODO optimization: implement via lerp: https://devblogs.nvidia.com/lerp-faster-cuda/
                for c in itertools.product(rounding_functions, repeat=field.spatial_dimensions):
                    weight = sp.Mul(*[1 - sp.Abs(f(offset) - offset) for (f, offset) in zip(c, offsets)])
                    index = [f(offset) for (f, offset) in zip(c, offsets)]
                    # Hardware boundary handling on GPU
                    if use_textures:
                        weight = sp.Mul(*[1 - sp.Abs(f(offset) - offset) for (f, offset) in zip(c, offsets)])
                        sum[channel_idx] += \
                            weight * absolute_access(index, channel_idx if field.index_dimensions else ())
                    # else boundary handling using software
                    elif str(self.interpolator.address_mode).lower() == 'border':
                        is_inside_field = sp.And(
                            *itertools.chain([i >= 0 for i in index],
                                             [idx < field.shape[dim] for (dim, idx) in enumerate(index)]))
                        index = [cast_func(i, default_int_type) for i in index]
                        sum[channel_idx] += sp.Piecewise(
                            (weight * absolute_access(index, channel_idx if field.index_dimensions else ()),
                                is_inside_field),
                            (sp.simplify(0), True)
                        )
                    elif str(self.interpolator.address_mode).lower() == 'clamp':
                        index = [sp.Min(sp.Max(0, cast_func(i, default_int_type)), field.spatial_shape[dim] - 1)
                                 for (dim, i) in enumerate(index)]
                        sum[channel_idx] += weight * \
                            absolute_access(index, channel_idx if field.index_dimensions else ())
                    elif str(self.interpolator.address_mode).lower() == 'wrap':
                        index = [sp.Mod(cast_func(i, default_int_type), field.shape[dim] - 1)
                                 for (dim, i) in enumerate(index)]
                        index = [cast_func(sp.Piecewise((i, i > 0),
                                                        (sp.Abs(cast_func(field.shape[dim] - 1 + i, default_int_type)),
                                                            True)), default_int_type)
                                 for (dim, i) in enumerate(index)]
                        sum[channel_idx] += weight * \
                            absolute_access(index, channel_idx if field.index_dimensions else ())
                        # sum[channel_idx] = 0
                    elif str(self.interpolator.address_mode).lower() == 'mirror':
                        def triangle_fun(x, half_period):
                            saw_tooth = sp.Abs(cast_func(x, default_int_type)) % (
                                cast_func(2 * half_period, create_type('int32')))
                            return sp.Piecewise((saw_tooth, saw_tooth < half_period),
                                                (2 * half_period - 1 - saw_tooth, True))
                        index = [cast_func(triangle_fun(i, field.shape[dim]),
                                           default_int_type) for (dim, i) in enumerate(index)]
                        sum[channel_idx] += weight * \
                            absolute_access(index, channel_idx if field.index_dimensions else ())
                    else:
                        raise NotImplementedError()
            elif self.interpolation_mode == InterpolationMode.CUBIC_SPLINE:
                raise NotImplementedError("only works with HW interpolation for float32")

            sum = [sp.factor(s) for s in sum]

            if field.index_dimensions:
                return sp.Matrix(sum)
            else:
                return sum[0]

    # noinspection SpellCheckingInspection
    __xnew__ = staticmethod(__new_stage2__)
    # noinspection SpellCheckingInspection
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

##########################################################################################
# GPU-specific fast specializations (for precision GPUs can also use above nodes/symbols #
##########################################################################################


class TextureCachedField:

    def __init__(self, parent_field,
                 address_mode=None,
                 filter_mode=None,
                 interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
                 use_normalized_coordinates=False,
                 read_as_integer=False
                 ):
        if isinstance(address_mode, str):
            address_mode = getattr(pycuda.driver.address_mode, address_mode.upper())

        if address_mode is None:
            address_mode = pycuda.driver.address_mode.BORDER
        if filter_mode is None:
            filter_mode = pycuda.driver.filter_mode.LINEAR

        # self, field_name, field_type, dtype, layout, shape, strides
        self.field = parent_field
        self.address_mode = address_mode
        self.filter_mode = filter_mode
        self.read_as_integer = read_as_integer
        self.use_normalized_coordinates = use_normalized_coordinates
        self.interpolation_mode = interpolation_mode
        self.symbol = TypedSymbol(str(self), self.field.dtype.numpy_dtype)
        self.symbol.interpolator = self
        self.symbol.field = self.field
        self.required_global_declarations = [TextureDeclaration(self)]

        # assert str(self.field.dtype) != 'double', "CUDA does not support double textures!"
        # assert dtype_supports_textures(self.field.dtype), "CUDA only supports texture types with 32 bits or less"

    @classmethod
    def from_interpolator(cls, interpolator: LinearInterpolator):
        if (isinstance(interpolator, cls)
                or (hasattr(interpolator, 'allow_textures') and not interpolator.allow_textures)):
            return interpolator
        obj = cls(interpolator.field, interpolator.address_mode, interpolation_mode=interpolator.interpolation_mode)
        return obj

    def at(self, offset):
        return TextureAccess(self.symbol, *offset)

    def __getitem__(self, offset):
        return TextureAccess(self.symbol, *offset)

    def __str__(self):
        return '%s_texture_%s' % (self.field.name, self.reproducible_hash)

    def __repr__(self):
        return self.__str__()

    @property
    def _hashable_contents(self):
        return (type(self),
                self.address_mode,
                self.filter_mode,
                self.read_as_integer,
                self.interpolation_mode,
                self.use_normalized_coordinates)

    def __hash__(self):
        return hash(self._hashable_contents)

    @property
    def reproducible_hash(self):
        return _hash(str(self._hashable_contents).encode()).hexdigest()


class TextureAccess(InterpolatorAccess):
    def __new__(cls, texture_symbol, offsets, *args, **kwargs):
        obj = TextureAccess.__xnew_cached_(cls, texture_symbol, offsets, *args, **kwargs)
        return obj

    def __new_stage2__(self, symbol, *offsets):
        obj = super().__xnew__(self, symbol, *offsets)
        obj.required_global_declarations = symbol.interpolator.required_global_declarations
        obj.required_global_declarations[0]._symbols_defined.add(obj)
        return obj

    def __str__(self):
        return '%s_texture(%s)' % (self.interpolator.field.name, ','.join(str(o) for o in self.offsets))

    @property
    def texture(self):
        return self.interpolator

    # noinspection SpellCheckingInspection
    __xnew__ = staticmethod(__new_stage2__)
    # noinspection SpellCheckingInspection
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))


class TextureDeclaration(Node):
    """
    A global declaration of a texture. Visible both for device and host code.

    .. code:: cpp

        // This Node represents the following global declaration
        texture<float, cudaTextureType2D, cudaReadModeElementType> x_texture_5acc9fced7b0dc3e;

        __device__ kernel(...) {
            // kernel acceses x_texture_5acc9fced7b0dc3e with tex2d(...)
        }

        __host__ launch_kernel(...) {
            // Host needs to bind the texture
            cudaBindTexture(0, x_texture_5acc9fced7b0dc3e, buffer, N*sizeof(float));
        }

    This has been deprecated by CUDA in favor of :class:`.TextureObject`.
    But texture objects are not yet supported by PyCUDA (https://github.com/inducer/pycuda/pull/174)
    """

    def __init__(self, parent_texture):
        self.texture = parent_texture
        self._symbols_defined = {self.texture.symbol}

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        return self._symbols_defined

    @property
    def args(self) -> Set[sp.Symbol]:
        return set()

    @property
    def headers(self):
        return ['"pycuda-helpers.hpp"']

    def __str__(self):
        from pystencils.backends.cuda_backend import CudaBackend
        return CudaBackend()(self)


class TextureObject(TextureDeclaration):
    """
    A CUDA texture object. Opposed to :class:`.TextureDeclaration` it is not declared globally but
    used as a function argument for the kernel call.

    Like :class:`.TextureDeclaration` it defines :class:`.TextureAccess` symbols.
    Just the printing representation is a bit different.
    """
    pass


def dtype_supports_textures(dtype):
    """
    Returns whether CUDA natively supports texture fetches with this numpy dtype.

    The maximum word size for a texture fetch is four bytes.

    With this trick also larger dtypes can be fetched:
    https://github.com/inducer/pycuda/blob/master/pycuda/cuda/pycuda-helpers.hpp

    """
    if hasattr(dtype, 'numpy_dtype'):
        dtype = dtype.numpy_dtype

    if isinstance(dtype, type):
        return dtype().itemsize <= 4

    return dtype.itemsize <= 4
