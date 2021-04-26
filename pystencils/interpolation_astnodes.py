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


class _InterpolationSymbol(TypedSymbol):

    def __new__(cls, name, field, interpolator):
        obj = cls.__xnew_cached_(cls, name, field, interpolator)
        return obj

    def __new_stage2__(cls, name, field, interpolator):
        obj = super().__xnew__(cls, name, 'dummy_symbol_carrying_field' + field.name)
        obj.field = field
        obj.interpolator = interpolator
        return obj

    def __getnewargs__(self):
        return self.name, self.field, self.interpolator

    def __getnewargs_ex__(self):
        return (self.name, self.field, self.interpolator), {}

    # noinspection SpellCheckingInspection
    __xnew__ = staticmethod(__new_stage2__)
    # noinspection SpellCheckingInspection
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))


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

        self.field = parent_field
        self.field.field_type = pystencils.field.FieldType.CUSTOM
        self.address_mode = address_mode
        self.use_normalized_coordinates = use_normalized_coordinates
        self.interpolation_mode = interpolation_mode
        self.hash_str = hashlib.md5(
            f'{self.field}_{address_mode}_{self.field.dtype}_{interpolation_mode}'.encode()).hexdigest()
        self.symbol = _InterpolationSymbol(str(self), parent_field, self)
        self.allow_textures = allow_textures

    @property
    def ndim(self):
        return self.field.ndim

    @property
    def _hashable_contents(self):
        return (str(self.address_mode),
                str(type(self)),
                self.hash_str,
                self.use_normalized_coordinates)

    def at(self, offset):
        return InterpolatorAccess(self.symbol, *[sp.S(o) for o in offset])

    def __getitem__(self, offset):
        return InterpolatorAccess(self.symbol, *[sp.S(o) for o in offset])

    def __str__(self):
        return f'{self.field.name}_interpolator_{self.reproducible_hash}'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self._hashable_contents)

    def __eq__(self, other):
        return hash(self) == hash(other)

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
    def __new__(cls, field, *offsets):
        obj = InterpolatorAccess.__xnew_cached_(cls, field, *offsets)
        return obj

    def __new_stage2__(cls, symbol, *offsets):
        assert offsets is not None
        obj = super().__xnew__(cls, '%s_interpolator_%s' %
                               (symbol.field.name, _hash(str(tuple(offsets)).encode()).hexdigest()),
                               symbol.field.dtype)
        obj.offsets = offsets
        obj.symbol = symbol
        obj.field = symbol.field
        obj.interpolator = symbol.interpolator
        return obj

    def _hashable_contents(self):
        return super()._hashable_content() + ((self.symbol, self.field, tuple(self.offsets), self.symbol.interpolator))

    def __str__(self):
        return f"{self.field.name}_interpolator({', '.join(str(o) for o in self.offsets)})"

    def __repr__(self):
        return self.__str__()

    def _latex(self, printer, *_):
        n = self.field.latex_name if self.field.latex_name else self.field.name
        foo = ", ".join(str(printer.doprint(o)) for o in self.offsets)
        return f'{n}_{{interpolator}}\\left({foo}\\right)'

    @property
    def ndim(self):
        return len(self.offsets)

    @property
    def is_texture(self):
        return isinstance(self.interpolator, TextureCachedField)

    def atoms(self, *types):
        if self.offsets:
            offsets = set(o for o in self.offsets if isinstance(o, types))
            if isinstance(self, *types):
                offsets.update([self])
            for o in self.offsets:
                if hasattr(o, 'atoms'):
                    offsets.update(set(o.atoms(*types)))
            return offsets
        else:
            return set()

    def neighbor(self, coord_id, offset):
        offset_list = list(self.offsets)
        offset_list[coord_id] += offset
        return self.interpolator.at(tuple(offset_list))

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
    def required_global_declarations(self):
        required_global_declarations = self.symbol.interpolator.required_global_declarations
        if required_global_declarations:
            required_global_declarations[0]._symbols_defined.add(self)
        return required_global_declarations

    @property
    def args(self):
        return [self.symbol, *self.offsets]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        return {self}

    @property
    def interpolation_mode(self):
        return self.interpolator.interpolation_mode

    @property
    def _diff_interpolation_vec(self):
        return sp.Matrix([DiffInterpolatorAccess(self.symbol, i, *self.offsets)
                          for i in range(len(self.offsets))])

    def diff(self, *symbols, **kwargs):
        if symbols == (self,):
            return 1
        rtn = self._diff_interpolation_vec.T * sp.Matrix(self.offsets).diff(*symbols, **kwargs)
        if rtn.shape == (1, 1):
            rtn = rtn[0, 0]
        return rtn

    def implementation_with_stencils(self):
        field = self.field

        default_int_type = create_type('int64')
        use_textures = isinstance(self.interpolator, TextureCachedField)
        if use_textures:
            def absolute_access(x, _):
                return self.symbol.interpolator.at((o for o in x))
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
                            saw_tooth = cast_func(sp.Abs(cast_func(x, 'int32')), 'int32') % (
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

    def __getnewargs__(self):
        return (self.symbol, *self.offsets)

    def __getnewargs_ex__(self):
        return (self.symbol, *self.offsets), {}


class DiffInterpolatorAccess(InterpolatorAccess):
    def __new__(cls, symbol, diff_coordinate_idx, *offsets):
        if symbol.interpolator.interpolation_mode == InterpolationMode.LINEAR:
            from pystencils.fd import Diff, Discretization2ndOrder
            return Discretization2ndOrder(1)(Diff(symbol.interpolator.at(offsets), diff_coordinate_idx))
        obj = DiffInterpolatorAccess.__xnew_cached_(cls, symbol, diff_coordinate_idx, *offsets)
        return obj

    def __new_stage2__(self, symbol: sp.Symbol, diff_coordinate_idx, *offsets):
        assert offsets is not None
        obj = super().__xnew__(self, symbol, *offsets)
        obj.diff_coordinate_idx = diff_coordinate_idx
        return obj

    def __hash__(self):
        return hash((self.symbol, self.field, self.diff_coordinate_idx, tuple(self.offsets), self.interpolator))

    def __str__(self):
        return '%s_diff%i_interpolator(%s)' % (self.field.name, self.diff_coordinate_idx,
                                               ', '.join(str(o) for o in self.offsets))

    def __repr__(self):
        return str(self)

    @property
    def args(self):
        return [self.symbol, self.diff_coordinate_idx, *self.offsets]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        return {self}

    @property
    def interpolation_mode(self):
        return self.interpolator.interpolation_mode

    # noinspection SpellCheckingInspection
    __xnew__ = staticmethod(__new_stage2__)
    # noinspection SpellCheckingInspection
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __getnewargs__(self):
        return (self.symbol, self.diff_coordinate_idx, *self.offsets)

    def __getnewargs_ex__(self):
        return (self.symbol, self.diff_coordinate_idx, *self.offsets), {}


##########################################################################################
# GPU-specific fast specializations (for precision GPUs can also use above nodes/symbols #
##########################################################################################


class TextureCachedField(Interpolator):

    def __init__(self, parent_field,
                 address_mode=None,
                 filter_mode=None,
                 interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
                 use_normalized_coordinates=False,
                 read_as_integer=False
                 ):
        super().__init__(parent_field, interpolation_mode, address_mode, use_normalized_coordinates)

        if address_mode is None:
            address_mode = 'border'
        if filter_mode is None:
            filter_mode = pycuda.driver.filter_mode.LINEAR

        self.read_as_integer = read_as_integer
        self.required_global_declarations = [TextureDeclaration(self)]

    @property
    def ndim(self):
        return self.field.ndim

    @classmethod
    def from_interpolator(cls, interpolator: LinearInterpolator):
        if (isinstance(interpolator, cls)
                or (hasattr(interpolator, 'allow_textures') and not interpolator.allow_textures)):
            return interpolator
        obj = cls(interpolator.field, interpolator.address_mode, interpolation_mode=interpolator.interpolation_mode)
        return obj

    def __str__(self):
        return f'{self.field.name}_texture_{self.reproducible_hash}'

    def __repr__(self):
        return self.__str__()

    @property
    def reproducible_hash(self):
        return _hash(str(self._hashable_contents).encode()).hexdigest()


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
        headers = ['"pycuda-helpers.hpp"']
        if self.texture.interpolation_mode == InterpolationMode.CUBIC_SPLINE:
            headers.append('"cubicTex%iD.cu"' % self.texture.ndim)
        return headers

    def __str__(self):
        from pystencils.backends.cuda_backend import CudaBackend
        return CudaBackend()(self)

    def __repr__(self):
        return str(self)


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
