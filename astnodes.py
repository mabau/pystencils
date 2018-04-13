import sympy as sp
from sympy.tensor import IndexedBase
from pystencils.field import Field
from pystencils.data_types import TypedSymbol, create_type, cast_func
from pystencils.sympyextensions import fast_subs
from typing import List, Set, Optional, Union, Any

NodeOrExpr = Union['Node', sp.Expr]


class Node(object):
    """Base class for all AST nodes."""

    def __init__(self, parent: Optional['Node'] = None):
        self.parent = parent

    @property
    def args(self) -> List[NodeOrExpr]:
        """Returns all arguments/children of this node."""
        return []

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        """Set of symbols which are defined by this node."""
        return set()

    @property
    def undefined_symbols(self) -> Set[sp.Symbol]:
        """Symbols which are used but are not defined inside this node."""
        raise NotImplementedError()

    def subs(self, *args, **kwargs) -> None:
        """Inplace! substitute, similar to sympy's but modifies the AST inplace."""
        for a in self.args:
            a.subs(*args, **kwargs)

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

        assert condition_expr.is_Boolean or condition_expr.is_Relational
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

    def subs(self, *args, **kwargs):
        self.true_block.subs(*args, **kwargs)
        if self.false_block:
            self.false_block.subs(*args, **kwargs)
        self.condition_expr = self.condition_expr.subs(*args, **kwargs)

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
        result.update(self.condition_expr.atoms(sp.Symbol))
        return result

    def __str__(self):
        return 'if:({!s}) '.format(self.condition_expr)

    def __repr__(self):
        return 'if:({!r}) '.format(self.condition_expr)


class KernelFunction(Node):

    class Argument:
        def __init__(self, name, dtype, symbol, kernel_function_node):
            from pystencils.transformations import symbol_name_to_variable_name
            self.name = name
            self.dtype = dtype
            self.is_field_ptr_argument = False
            self.is_field_shape_argument = False
            self.is_field_stride_argument = False
            self.is_field_argument = False
            self.field_name = ""
            self.coordinate = None
            self.symbol = symbol

            if name.startswith(Field.DATA_PREFIX):
                self.is_field_ptr_argument = True
                self.is_field_argument = True
                self.field_name = name[len(Field.DATA_PREFIX):]
            elif name.startswith(Field.SHAPE_PREFIX):
                self.is_field_shape_argument = True
                self.is_field_argument = True
                self.field_name = name[len(Field.SHAPE_PREFIX):]
            elif name.startswith(Field.STRIDE_PREFIX):
                self.is_field_stride_argument = True
                self.is_field_argument = True
                self.field_name = name[len(Field.STRIDE_PREFIX):]

            self.field = None
            if self.is_field_argument:
                field_map = {symbol_name_to_variable_name(f.name): f for f in kernel_function_node.fields_accessed}
                self.field = field_map[self.field_name]

        def __lt__(self, other):
            def score(l):
                if l.is_field_ptr_argument:
                    return -4
                elif l.is_field_shape_argument:
                    return -3
                elif l.is_field_stride_argument:
                    return -2
                return 0

            if score(self) < score(other):
                return True
            elif score(self) == score(other):
                return self.name < other.name
            else:
                return False

        def __repr__(self):
            return '<{0} {1}>'.format(self.dtype, self.name)

    def __init__(self, body, ghost_layers=None, function_name="kernel", backend=""):
        super(KernelFunction, self).__init__()
        self._body = body
        body.parent = self
        self._parameters = None
        self.function_name = function_name
        self._body.parent = self
        self.compile = None
        self.ghost_layers = ghost_layers
        # these variables are assumed to be global, so no automatic parameter is generated for them
        self.global_variables = set()
        self.backend = backend

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    @property
    def parameters(self):
        self._update_parameters()
        return self._parameters

    @property
    def body(self):
        return self._body

    @property
    def args(self):
        return [self._body]

    @property
    def fields_accessed(self):
        """Set of Field instances: fields which are accessed inside this kernel function"""
        return set(o.field for o in self.atoms(ResolvedFieldAccess))

    def _update_parameters(self):
        undefined_symbols = self._body.undefined_symbols - self.global_variables
        self._parameters = [KernelFunction.Argument(s.name, s.dtype, s, self) for s in undefined_symbols]

        self._parameters.sort()

    def __str__(self):
        self._update_parameters()
        return '{0} {1}({2})\n{3}'.format(type(self).__name__, self.function_name, self.parameters,
                                          ("\t" + "\t".join(str(self.body).splitlines(True))))

    def __repr__(self):
        self._update_parameters()
        return '{0} {1}({2})'.format(type(self).__name__, self.function_name, self.parameters)


class Block(Node):
    def __init__(self, nodes: List[Node]):
        super(Block, self).__init__()
        self._nodes = nodes
        self.parent = None
        for n in self._nodes:
            n.parent = self

    @property
    def args(self):
        return self._nodes

    def insert_front(self, node):
        node.parent = self
        self._nodes.insert(0, node)

    def insert_before(self, new_node, insert_before):
        new_node.parent = self
        idx = self._nodes.index(insert_before)

        # move all assignment (definitions to the top)
        if isinstance(new_node, SympyAssignment) and new_node.is_declaration:
            while idx > 0:
                pn = self._nodes[idx - 1]
                if isinstance(pn, LoopOverCoordinate) or isinstance(pn, Conditional):
                    idx -= 1
                else:
                    break
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
            result.update(a.symbols_defined)
        return result

    @property
    def undefined_symbols(self):
        result = set()
        defined_symbols = set()
        for a in self.args:
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

    def __init__(self, body, coordinate_to_loop_over, start, stop, step=1):
        super(LoopOverCoordinate, self).__init__(parent=None)
        self.body = body
        body.parent = self
        self.coordinate_to_loop_over = coordinate_to_loop_over
        self.start = start
        self.stop = stop
        self.step = step
        self.body.parent = self
        self.prefix_lines = []

    def new_loop_with_different_body(self, new_body):
        result = LoopOverCoordinate(new_body, self.coordinate_to_loop_over, self.start, self.stop, self.step)
        result.prefix_lines = [l for l in self.prefix_lines]
        return result

    def subs(self, *args, **kwargs):
        self.body.subs(*args, **kwargs)
        if hasattr(self.start, "subs"):
            self.start = self.start.subs(*args, **kwargs)
        if hasattr(self.stop, "subs"):
            self.stop = self.stop.subs(*args, **kwargs)
        if hasattr(self.step, "subs"):
            self.step = self.step.subs(*args, **kwargs)

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
        return "%s_%s" % (LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX, coordinate_to_loop_over)

    @property
    def loop_counter_name(self):
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
        return TypedSymbol(LoopOverCoordinate.get_loop_counter_name(coordinate_to_loop_over), 'int')

    @property
    def loop_counter_symbol(self):
        return LoopOverCoordinate.get_loop_counter_symbol(self.coordinate_to_loop_over)

    @property
    def is_outermost_loop(self):
        from pystencils.transformations import get_next_parent_of_type
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
    def __init__(self, lhs_symbol, rhs_expr, is_const=True):
        super(SympyAssignment, self).__init__(parent=None)
        self._lhs_symbol = lhs_symbol
        self.rhs = rhs_expr
        self._is_declaration = True
        is_cast = self._lhs_symbol.func == cast_func
        if isinstance(self._lhs_symbol, Field.Access) or isinstance(self._lhs_symbol, ResolvedFieldAccess) or is_cast:
            self._is_declaration = False
        self._is_const = is_const

    @property
    def lhs(self):
        return self._lhs_symbol

    @lhs.setter
    def lhs(self, new_value):
        self._lhs_symbol = new_value
        self._is_declaration = True
        is_cast = self._lhs_symbol.func == cast_func
        if isinstance(self._lhs_symbol, Field.Access) or isinstance(self._lhs_symbol, sp.Indexed) or is_cast:
            self._is_declaration = False

    def subs(self, *args, **kwargs):
        self.lhs = fast_subs(self.lhs, *args, **kwargs)
        self.rhs = fast_subs(self.rhs, *args, **kwargs)

    @property
    def args(self):
        return [self._lhs_symbol, self.rhs]

    @property
    def symbols_defined(self):
        if not self._is_declaration:
            return set()
        return {self._lhs_symbol}

    @property
    def undefined_symbols(self):
        result = self.rhs.atoms(sp.Symbol)
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
            raise ValueError('%s is not in args of %s' % (replacement, self.__class__))

    def __repr__(self):
        return repr(self.lhs) + " = " + repr(self.rhs)

    def _repr_html_(self):
        printed_lhs = sp.latex(self.lhs)
        printed_rhs = sp.latex(self.rhs)
        return f"${printed_lhs} = {printed_rhs}$"


class ResolvedFieldAccess(sp.Indexed):
    def __new__(cls, base, linearized_index, field, offsets, idx_coordinate_values):
        if not isinstance(base, IndexedBase):
            base = IndexedBase(base, shape=(1,))
        obj = super(ResolvedFieldAccess, cls).__new__(cls, base, linearized_index)
        obj.field = field
        obj.offsets = offsets
        obj.idx_coordinate_values = idx_coordinate_values
        return obj

    def _eval_subs(self, old, new):
        return ResolvedFieldAccess(self.args[0],
                                   self.args[1].subs(old, new),
                                   self.field, self.offsets, self.idx_coordinate_values)

    def fast_subs(self, substitutions):
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
        return "%s (%s)" % (top, self.typed_symbol.dtype)

    def __getnewargs__(self):
        return self.base, self.indices[0], self.field, self.offsets, self.idx_coordinate_values


class TemporaryMemoryAllocation(Node):
    def __init__(self, typed_symbol, size):
        super(TemporaryMemoryAllocation, self).__init__(parent=None)
        self.symbol = typed_symbol
        self.size = size

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


class TemporaryMemoryFree(Node):
    def __init__(self, typed_symbol):
        super(TemporaryMemoryFree, self).__init__(parent=None)
        self.symbol = typed_symbol

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    @property
    def args(self):
        return []
