import collections.abc
import itertools
import uuid
from typing import Any, List, Optional, Sequence, Set, Union

import sympy as sp

import pystencils
from pystencils.typing.utilities import create_type, get_next_parent_of_type
from pystencils.enums import Target, Backend
from pystencils.field import Field
from pystencils.typing.typed_sympy import FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol, TypedSymbol
from pystencils.sympyextensions import fast_subs

NodeOrExpr = Union['Node', sp.Expr]


class Node:
    """Base class for all AST nodes."""

    def __init__(self, parent: Optional['Node'] = None):
        self.parent = parent

    @property
    def args(self) -> List[NodeOrExpr]:
        """Returns all arguments/children of this node."""
        raise NotImplementedError()

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        """Set of symbols which are defined by this node."""
        raise NotImplementedError()

    @property
    def undefined_symbols(self) -> Set[sp.Symbol]:
        """Symbols which are used but are not defined inside this node."""
        raise NotImplementedError()

    def subs(self, subs_dict) -> None:
        """Inplace! Substitute, similar to sympy's but modifies the AST inplace."""
        for i, a in enumerate(self.args):
            result = a.subs(subs_dict)
            if isinstance(a, sp.Expr):  # sympy expressions' subs is out-of-place
                self.args[i] = result
            else:  # all other should be in-place
                assert result is None

    @property
    def func(self):
        return self.__class__

    def atoms(self, arg_type) -> Set[Any]:
        """Returns a set of all descendants recursively, which are an instance of the given type."""
        result = set()
        for arg in self.args:
            if isinstance(arg, arg_type):
                result.add(arg)
            result.update(arg.atoms(arg_type))
        return result


class Conditional(Node):
    """Conditional that maps to a 'if' statement in C/C++.

    Try to avoid using this node inside of loops, since currently this construction can not be vectorized.
    Consider using assignments with sympy.Piecewise in this case.

    Args:
        condition_expr: sympy relational expression
        true_block: block which is run if conditional is true
        false_block: optional block which is run if conditional is false
    """

    def __init__(self, condition_expr: sp.Basic, true_block: Union['Block', 'SympyAssignment'],
                 false_block: Optional['Block'] = None) -> None:
        super(Conditional, self).__init__(parent=None)

        self.condition_expr = condition_expr

        def handle_child(c):
            if c is None:
                return None
            if not isinstance(c, Block):
                c = Block([c])
            c.parent = self
            return c

        self.true_block = handle_child(true_block)
        self.false_block = handle_child(false_block)

    def subs(self, subs_dict):
        self.true_block.subs(subs_dict)
        if self.false_block:
            self.false_block.subs(subs_dict)
        self.condition_expr = self.condition_expr.subs(subs_dict)

    @property
    def args(self):
        result = [self.condition_expr, self.true_block]
        if self.false_block:
            result.append(self.false_block)
        return result

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        result = self.true_block.undefined_symbols
        if self.false_block:
            result.update(self.false_block.undefined_symbols)
        if hasattr(self.condition_expr, 'atoms'):
            result.update(self.condition_expr.atoms(sp.Symbol))
        return result

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        result = f'if:({self.condition_expr!r}) '
        if self.true_block:
            result += f'\n\t{self.true_block}) '
        if self.false_block:
            result = 'else: '
            result += f'\n\t{self.false_block} '

        return result

    def replace_by_true_block(self):
        """Replaces the conditional by its True block"""
        self.parent.replace(self, [self.true_block])

    def replace_by_false_block(self):
        """Replaces the conditional by its False block"""
        self.parent.replace(self, [self.false_block] if self.false_block else [])


class KernelFunction(Node):
    class Parameter:
        """Function parameter.

        Each undefined symbol in a `KernelFunction` node becomes a parameter to the function.
        Parameters are either symbols introduced by the user that never occur on the left hand side of an
        Assignment, or are related to fields/arrays passed to the function.

        A parameter consists of the typed symbol (symbol property). For field related parameters this is a symbol
        defined in pystencils.kernelparameters.
        If the parameter is related to one or multiple fields, these fields are referenced in the fields property.
        """

        def __init__(self, symbol, fields):
            self.symbol = symbol  # type: TypedSymbol
            self.fields = fields  # type: Sequence[Field]

        def __repr__(self):
            return repr(self.symbol)

        @property
        def is_field_stride(self):
            return isinstance(self.symbol, FieldStrideSymbol)

        @property
        def is_field_shape(self):
            return isinstance(self.symbol, FieldShapeSymbol)

        @property
        def is_field_pointer(self):
            return isinstance(self.symbol, FieldPointerSymbol)

        @property
        def is_field_parameter(self):
            return self.is_field_pointer or self.is_field_shape or self.is_field_stride

        @property
        def field_name(self):
            return self.fields[0].name

    def __init__(self, body, target: Target, backend: Backend, compile_function, ghost_layers,
                 function_name: str = "kernel",
                 assignments=None):
        super(KernelFunction, self).__init__()
        self._body = body
        body.parent = self
        self.function_name = function_name
        self._body.parent = self
        self.ghost_layers = ghost_layers
        self._target = target
        self._backend = backend
        # these variables are assumed to be global, so no automatic parameter is generated for them
        self.global_variables = set()
        self.instruction_set = None  # used in `vectorize` function to tell the backend which i.s. (SSE,AVX) to use
        # function that compiles the node to a Python callable, is set by the backends
        self._compile_function = compile_function
        self.assignments = assignments
        # If nontemporal stores are activated together with the Neon instruction set it results in cacheline zeroing
        # For cacheline zeroing the information of the field size for each field is needed. Thus, in this case
        # all field sizes are kernel parameters and not just the common field size used for the loops
        self.use_all_written_field_sizes = False

    @property
    def target(self):
        """See pystencils.Target"""
        return self._target

    @property
    def backend(self):
        """Backend for generating the code: `Backend`"""
        return self._backend

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value):
        self._body = value
        self._body.parent = self

    @property
    def args(self):
        return self._body,

    @property
    def fields_accessed(self) -> Set[Field]:
        """Set of Field instances: fields which are accessed inside this kernel function"""
        return set(o.field for o in itertools.chain(self.atoms(ResolvedFieldAccess)))

    @property
    def fields_written(self) -> Set[Field]:
        assignments = self.atoms(SympyAssignment)
        return set().union(itertools.chain.from_iterable([f.field for f in a.lhs.free_symbols if hasattr(f, 'field')]
                                                         for a in assignments))

    @property
    def fields_read(self) -> Set[Field]:
        assignments = self.atoms(SympyAssignment)
        return set().union(itertools.chain.from_iterable([f.field for f in a.rhs.free_symbols if hasattr(f, 'field')]
                                                         for a in assignments))

    def get_parameters(self) -> Sequence['KernelFunction.Parameter']:
        """Returns list of parameters for this function.

        This function is expensive, cache the result where possible!
        """
        field_map = {f.name: f for f in self.fields_accessed}
        sizes = set()

        if self.use_all_written_field_sizes:
            sizes = set().union(*(a.shape[:a.spatial_dimensions] for a in self.fields_written))
            sizes = filter(lambda s: isinstance(s, FieldShapeSymbol), sizes)

        def get_fields(symbol):
            if hasattr(symbol, 'field_name'):
                return field_map[symbol.field_name],
            elif hasattr(symbol, 'field_names'):
                return tuple(field_map[fn] for fn in symbol.field_names)
            return ()

        argument_symbols = self._body.undefined_symbols - self.global_variables
        argument_symbols.update(sizes)
        parameters = [self.Parameter(symbol, get_fields(symbol)) for symbol in argument_symbols]
        if hasattr(self, 'indexing'):
            parameters += [self.Parameter(s, []) for s in self.indexing.symbolic_parameters()]
        parameters.sort(key=lambda p: p.symbol.name)
        return parameters

    def __str__(self):
        params = [p.symbol for p in self.get_parameters()]
        return '{0} {1}({2})\n{3}'.format(type(self).__name__, self.function_name, params,
                                          ("\t" + "\t".join(str(self.body).splitlines(True))))

    def __repr__(self):
        params = [p.symbol for p in self.get_parameters()]
        return f'{type(self).__name__} {self.function_name}({params})'

    def compile(self, *args, **kwargs):
        if self._compile_function is None:
            raise ValueError("No compile-function provided for this KernelFunction node")
        return self._compile_function(self, *args, **kwargs)


class SkipIteration(Node):
    @property
    def args(self):
        return []

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()


class Block(Node):
    def __init__(self, nodes: List[Node]):
        super(Block, self).__init__()
        if not isinstance(nodes, list):
            nodes = [nodes]
        self._nodes = nodes
        self.parent = None
        for n in self._nodes:
            try:
                n.parent = self
            except AttributeError:
                pass

    @property
    def args(self):
        return self._nodes

    def subs(self, subs_dict) -> None:
        for a in self.args:
            a.subs(subs_dict)

    def fast_subs(self, subs_dict, skip=None):
        self._nodes = [fast_subs(a, subs_dict, skip) for a in self._nodes]
        return self

    def insert_front(self, node, if_not_exists=False):
        if if_not_exists and len(self._nodes) > 0 and self._nodes[0] == node:
            return
        if isinstance(node, collections.abc.Iterable):
            node = list(node)
            for n in node:
                n.parent = self

            self._nodes = node + self._nodes
        else:
            node.parent = self
            self._nodes.insert(0, node)

    def insert_before(self, new_node, insert_before, if_not_exists=False):
        new_node.parent = self
        assert self._nodes.count(insert_before) == 1
        idx = self._nodes.index(insert_before)

        # move all assignment (definitions to the top)
        if isinstance(new_node, SympyAssignment) and new_node.is_declaration:
            while idx > 0:
                pn = self._nodes[idx - 1]
                if isinstance(pn, LoopOverCoordinate) or isinstance(pn, Conditional):
                    idx -= 1
                else:
                    break
        if not if_not_exists or self._nodes[idx] != new_node:
            self._nodes.insert(idx, new_node)

    def insert_after(self, new_node, insert_after, if_not_exists=False):
        new_node.parent = self
        assert self._nodes.count(insert_after) == 1
        idx = self._nodes.index(insert_after) + 1

        # move all assignment (definitions to the top)
        if isinstance(new_node, SympyAssignment) and new_node.is_declaration:
            while idx > 0:
                pn = self._nodes[idx - 1]
                if isinstance(pn, LoopOverCoordinate) or isinstance(pn, Conditional):
                    idx -= 1
                else:
                    break
        if not if_not_exists or not (self._nodes[idx - 1] == new_node
                                     or (idx < len(self._nodes) and self._nodes[idx] == new_node)):
            self._nodes.insert(idx, new_node)

    def append(self, node):
        if isinstance(node, list) or isinstance(node, tuple):
            for n in node:
                n.parent = self
                self._nodes.append(n)
        else:
            node.parent = self
            self._nodes.append(node)

    def take_child_nodes(self):
        tmp = self._nodes
        self._nodes = []
        return tmp

    def replace(self, child, replacements):
        assert self._nodes.count(child) == 1
        idx = self._nodes.index(child)
        del self._nodes[idx]
        if type(replacements) is list:
            for e in replacements:
                e.parent = self
            self._nodes = self._nodes[:idx] + replacements + self._nodes[idx:]
        else:
            replacements.parent = self
            self._nodes.insert(idx, replacements)

    @property
    def symbols_defined(self):
        result = set()
        for a in self.args:
            if isinstance(a, pystencils.Assignment):
                result.update(a.free_symbols)
            else:
                result.update(a.symbols_defined)
        return result

    @property
    def undefined_symbols(self):
        result = set()
        defined_symbols = set()
        for a in self.args:
            if isinstance(a, pystencils.Assignment):
                result.update(a.free_symbols)
                defined_symbols.update({a.lhs})
            else:
                result.update(a.undefined_symbols)
                defined_symbols.update(a.symbols_defined)
        return result - defined_symbols

    def __str__(self):
        return "Block " + ''.join('{!s}\n'.format(node) for node in self._nodes)

    def __repr__(self):
        return "Block"


class PragmaBlock(Block):
    def __init__(self, pragma_line, nodes):
        super(PragmaBlock, self).__init__(nodes)
        self.pragma_line = pragma_line
        for n in nodes:
            n.parent = self

    def __repr__(self):
        return self.pragma_line


class LoopOverCoordinate(Node):
    LOOP_COUNTER_NAME_PREFIX = "ctr"
    BLOCK_LOOP_COUNTER_NAME_PREFIX = "_blockctr"

    def __init__(self, body, coordinate_to_loop_over, start, stop, step=1, is_block_loop=False):
        super(LoopOverCoordinate, self).__init__(parent=None)
        self.body = body
        body.parent = self
        self.coordinate_to_loop_over = coordinate_to_loop_over
        self.start = start
        self.stop = stop
        self.step = step
        self.body.parent = self
        self.prefix_lines = []
        self.is_block_loop = is_block_loop

    def new_loop_with_different_body(self, new_body):
        result = LoopOverCoordinate(new_body, self.coordinate_to_loop_over, self.start, self.stop,
                                    self.step, self.is_block_loop)
        result.prefix_lines = [prefix_line for prefix_line in self.prefix_lines]
        return result

    def subs(self, subs_dict):
        self.body.subs(subs_dict)
        if hasattr(self.start, "subs"):
            self.start = self.start.subs(subs_dict)
        if hasattr(self.stop, "subs"):
            self.stop = self.stop.subs(subs_dict)
        if hasattr(self.step, "subs"):
            self.step = self.step.subs(subs_dict)

    def fast_subs(self, subs_dict, skip=None):
        self.body = fast_subs(self.body, subs_dict, skip)
        if isinstance(self.start, sp.Basic):
            self.start = fast_subs(self.start, subs_dict, skip)
        if isinstance(self.stop, sp.Basic):
            self.stop = fast_subs(self.stop, subs_dict, skip)
        if isinstance(self.step, sp.Basic):
            self.step = fast_subs(self.step, subs_dict, skip)
        return self

    @property
    def args(self):
        result = [self.body]
        for e in [self.start, self.stop, self.step]:
            if hasattr(e, "args"):
                result.append(e)
        return result

    def replace(self, child, replacement):
        if child == self.body:
            self.body = replacement
        elif child == self.start:
            self.start = replacement
        elif child == self.step:
            self.step = replacement
        elif child == self.stop:
            self.stop = replacement

    @property
    def symbols_defined(self):
        return {self.loop_counter_symbol}

    @property
    def undefined_symbols(self):
        result = self.body.undefined_symbols
        for possible_symbol in [self.start, self.stop, self.step]:
            if isinstance(possible_symbol, Node) or isinstance(possible_symbol, sp.Basic):
                result.update(possible_symbol.atoms(sp.Symbol))
        return result - {self.loop_counter_symbol}

    @staticmethod
    def get_loop_counter_name(coordinate_to_loop_over):
        return f"{LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX}_{coordinate_to_loop_over}"

    @staticmethod
    def get_block_loop_counter_name(coordinate_to_loop_over):
        return f"{LoopOverCoordinate.BLOCK_LOOP_COUNTER_NAME_PREFIX}_{coordinate_to_loop_over}"

    @property
    def loop_counter_name(self):
        if self.is_block_loop:
            return LoopOverCoordinate.get_block_loop_counter_name(self.coordinate_to_loop_over)
        else:
            return LoopOverCoordinate.get_loop_counter_name(self.coordinate_to_loop_over)

    @staticmethod
    def is_loop_counter_symbol(symbol):
        prefix = LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX
        if not symbol.name.startswith(prefix):
            return None
        if symbol.dtype != create_type('int'):
            return None
        coordinate = int(symbol.name[len(prefix) + 1:])
        return coordinate

    @staticmethod
    def get_loop_counter_symbol(coordinate_to_loop_over):
        return TypedSymbol(LoopOverCoordinate.get_loop_counter_name(coordinate_to_loop_over), 'int', nonnegative=True)

    @staticmethod
    def get_block_loop_counter_symbol(coordinate_to_loop_over):
        return TypedSymbol(LoopOverCoordinate.get_block_loop_counter_name(coordinate_to_loop_over),
                           'int',
                           nonnegative=True)

    @property
    def loop_counter_symbol(self):
        if self.is_block_loop:
            return self.get_block_loop_counter_symbol(self.coordinate_to_loop_over)
        else:
            return self.get_loop_counter_symbol(self.coordinate_to_loop_over)

    @property
    def is_outermost_loop(self):
        return get_next_parent_of_type(self, LoopOverCoordinate) is None

    @property
    def is_innermost_loop(self):
        return len(self.atoms(LoopOverCoordinate)) == 0

    def __str__(self):
        return 'for({!s}={!s}; {!s}<{!s}; {!s}+={!s})\n{!s}'.format(self.loop_counter_name, self.start,
                                                                    self.loop_counter_name, self.stop,
                                                                    self.loop_counter_name, self.step,
                                                                    ("\t" + "\t".join(str(self.body).splitlines(True))))

    def __repr__(self):
        return 'for({!s}={!s}; {!s}<{!s}; {!s}+={!s})'.format(self.loop_counter_name, self.start,
                                                              self.loop_counter_name, self.stop,
                                                              self.loop_counter_name, self.step)


class SympyAssignment(Node):
    def __init__(self, lhs_symbol, rhs_expr, is_const=True, use_auto=False):
        super(SympyAssignment, self).__init__(parent=None)
        self._lhs_symbol = sp.sympify(lhs_symbol)
        self.rhs = sp.sympify(rhs_expr)
        self._is_const = is_const
        self._is_declaration = self.__is_declaration()
        self.use_auto = use_auto

    def __is_declaration(self):
        from pystencils.typing import CastFunc
        if isinstance(self._lhs_symbol, CastFunc):
            return False
        if any(isinstance(self._lhs_symbol, c) for c in (Field.Access, sp.Indexed, TemporaryMemoryAllocation)):
            return False
        return True

    @property
    def lhs(self):
        return self._lhs_symbol

    @lhs.setter
    def lhs(self, new_value):
        self._lhs_symbol = new_value
        self._is_declaration = self.__is_declaration()

    def subs(self, subs_dict):
        self.lhs = fast_subs(self.lhs, subs_dict)
        self.rhs = fast_subs(self.rhs, subs_dict)

    def optimize(self, optimizations):
        try:
            from sympy.codegen.rewriting import optimize
            self.rhs = optimize(self.rhs, optimizations)
        except Exception:
            pass

    @property
    def args(self):
        return [self._lhs_symbol, self.rhs, sp.sympify(self._is_const)]

    @property
    def symbols_defined(self):
        if not self._is_declaration:
            return set()
        return {self._lhs_symbol}

    @property
    def undefined_symbols(self):
        result = {s for s in self.rhs.free_symbols if not isinstance(s, sp.Indexed)}
        # Add loop counters if there a field accesses
        loop_counters = set()
        for symbol in result:
            if isinstance(symbol, Field.Access):
                for i in range(len(symbol.offsets)):
                    loop_counters.add(LoopOverCoordinate.get_loop_counter_symbol(i))
        result.update(loop_counters)
        
        result.update(self._lhs_symbol.atoms(sp.Symbol))
        
        return result

    @property
    def is_declaration(self):
        return self._is_declaration

    @property
    def is_const(self):
        return self._is_const

    def replace(self, child, replacement):
        if child == self.lhs:
            replacement.parent = self
            self.lhs = replacement
        elif child == self.rhs:
            replacement.parent = self
            self.rhs = replacement
        else:
            raise ValueError(f'{replacement} is not in args of {self.__class__}')

    def __repr__(self):
        return repr(self.lhs) + " ← " + repr(self.rhs)

    def _repr_html_(self):
        printed_lhs = sp.latex(self.lhs)
        printed_rhs = sp.latex(self.rhs)
        return f"${printed_lhs} \\leftarrow {printed_rhs}$"

    def __hash__(self):
        return hash((self.lhs, self.rhs))

    def __eq__(self, other):
        return type(self) == type(other) and (self.lhs, self.rhs) == (other.lhs, other.rhs)


class ResolvedFieldAccess(sp.Indexed):
    def __new__(cls, base, linearized_index, field, offsets, idx_coordinate_values):
        if not isinstance(base, sp.IndexedBase):
            assert isinstance(base, TypedSymbol)
            base = sp.IndexedBase(base, shape=(1,))
            assert isinstance(base.label, TypedSymbol)
        obj = super(ResolvedFieldAccess, cls).__new__(cls, base, linearized_index)
        obj.field = field
        obj.offsets = offsets
        obj.idx_coordinate_values = idx_coordinate_values
        return obj

    def _eval_subs(self, old, new):
        return ResolvedFieldAccess(self.args[0],
                                   self.args[1].subs(old, new),
                                   self.field, self.offsets, self.idx_coordinate_values)

    def fast_subs(self, substitutions, skip=None):
        if self in substitutions:
            return substitutions[self]
        return ResolvedFieldAccess(self.args[0].subs(substitutions),
                                   self.args[1].subs(substitutions),
                                   self.field, self.offsets, self.idx_coordinate_values)

    def _hashable_content(self):
        super_class_contents = super(ResolvedFieldAccess, self)._hashable_content()
        return super_class_contents + tuple(self.offsets) + (repr(self.idx_coordinate_values), hash(self.field))

    @property
    def typed_symbol(self):
        return self.base.label

    def __str__(self):
        top = super(ResolvedFieldAccess, self).__str__()
        return f"{top} ({self.typed_symbol.dtype})"

    def __getnewargs__(self):
        return self.base, self.indices[0], self.field, self.offsets, self.idx_coordinate_values

    def __getnewargs_ex__(self):
        return (self.base, self.indices[0], self.field, self.offsets, self.idx_coordinate_values), {}


class TemporaryMemoryAllocation(Node):
    """Node for temporary memory buffer allocation.

    Always allocates aligned memory.

    Args:
        typed_symbol: symbol used as pointer (has to be typed)
        size: number of elements to allocate
        align_offset: the align_offset's element is aligned
    """

    def __init__(self, typed_symbol: TypedSymbol, size, align_offset):
        super(TemporaryMemoryAllocation, self).__init__(parent=None)
        self.symbol = typed_symbol
        self.size = size
        self.headers = ['<stdlib.h>']
        self._align_offset = align_offset

    @property
    def symbols_defined(self):
        return {self.symbol}

    @property
    def undefined_symbols(self):
        if isinstance(self.size, sp.Basic):
            return self.size.atoms(sp.Symbol)
        else:
            return set()

    @property
    def args(self):
        return [self.symbol]

    def offset(self, byte_alignment):
        """Number of ELEMENTS to skip for a pointer that is aligned to byte_alignment."""
        np_dtype = self.symbol.dtype.base_type.numpy_dtype
        assert byte_alignment % np_dtype.itemsize == 0
        return -self._align_offset % (byte_alignment / np_dtype.itemsize)


class TemporaryMemoryFree(Node):
    def __init__(self, alloc_node):
        super(TemporaryMemoryFree, self).__init__(parent=None)
        self.alloc_node = alloc_node

    @property
    def symbol(self):
        return self.alloc_node.symbol

    def offset(self, byte_alignment):
        return self.alloc_node.offset(byte_alignment)

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    @property
    def args(self):
        return []


def early_out(condition):
    from pystencils.cpu.vectorization import vec_all
    return Conditional(vec_all(condition), Block([SkipIteration()]))


def get_dummy_symbol(dtype='bool'):
    return TypedSymbol(f'dummy{uuid.uuid4().hex}', create_type(dtype))


class SourceCodeComment(Node):
    def __init__(self, text):
        self.text = text

    @property
    def args(self):
        return []

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    def __str__(self):
        return "/* " + self.text + " */"

    def __repr__(self):
        return self.__str__()


class EmptyLine(Node):
    def __init__(self):
        pass

    @property
    def args(self):
        return []

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    def __str__(self):
        return ""

    def __repr__(self):
        return self.__str__()


class ConditionalFieldAccess(sp.Function):
    """
    :class:`pystencils.Field.Access` that is only executed if a certain condition is met.
    Can be used, for instance, for out-of-bound checks.
    """

    def __new__(cls, field_access, outofbounds_condition, outofbounds_value=0):
        return sp.Function.__new__(cls, field_access, outofbounds_condition, sp.S(outofbounds_value))

    @property
    def access(self):
        return self.args[0]

    @property
    def outofbounds_condition(self):
        return self.args[1]

    @property
    def outofbounds_value(self):
        return self.args[2]

    def __getnewargs__(self):
        return self.access, self.outofbounds_condition, self.outofbounds_value

    def __getnewargs_ex__(self):
        return (self.access, self.outofbounds_condition, self.outofbounds_value), {}
