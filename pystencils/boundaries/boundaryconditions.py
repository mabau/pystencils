from typing import Any, List, Tuple

from pystencils.astnodes import SympyAssignment
from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo
from pystencils.typing import create_type


class Boundary:
    """Base class all boundaries should derive from"""

    inner_or_boundary = True
    single_link = False

    def __init__(self, name=None):
        self._name = name

    def __call__(self, field, direction_symbol, index_field) -> List[SympyAssignment]:
        """Defines the boundary behavior and must therefore be implemented by all boundaries.

        Here the boundary is defined as a list of sympy assignments, from which a boundary kernel is generated.

        Args:
            field: pystencils field where boundary condition should be applied.
                   The current cell is cell next to the boundary, which is influenced by the boundary
                   cell i.e. has a link from the boundary cell to itself.
            direction_symbol: a sympy symbol that can be used as index to the pdf_field. It describes
                              the direction pointing from the fluid to the boundary cell
            index_field: the boundary index field that can be used to retrieve and update boundary data
        """
        raise NotImplementedError("Boundary class has to overwrite __call__")

    @property
    def additional_data(self) -> Tuple[str, Any]:
        """Return a list of (name, type) tuples for additional data items required in this boundary
        These data items can either be initialized in separate kernel see additional_data_kernel_init or by
        Python callbacks - see additional_data_callback """
        return ()

    @property
    def additional_data_init_callback(self):
        """Return a callback function called with a boundary data setter object and returning a dict of
        data-name to data for each element that should be initialized"""
        return None

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return type(self).__name__

    @name.setter
    def name(self, new_value):
        self._name = new_value


class Neumann(Boundary):

    inner_or_boundary = False
    single_link = True

    def __call__(self, field, direction_symbol, **kwargs):

        neighbor = BoundaryOffsetInfo.offset_from_dir(direction_symbol, field.spatial_dimensions)
        if field.index_dimensions == 0:
            return [SympyAssignment(field.center, field[neighbor])]
        else:
            from itertools import product
            if not field.has_fixed_index_shape:
                raise NotImplementedError("Neumann boundary works only for fields with fixed index shape")
            index_iter = product(*(range(i) for i in field.index_shape))
            return [SympyAssignment(field(*idx), field[neighbor](*idx)) for idx in index_iter]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("Neumann")

    def __eq__(self, other):
        return type(other) == Neumann


class Dirichlet(Boundary):

    inner_or_boundary = False
    single_link = True

    def __init__(self, value, name=None):
        super().__init__(name)
        self._value = value

    @property
    def additional_data(self):
        if callable(self._value):
            return [('value', create_type("double"))]
        else:
            return []

    @property
    def additional_data_init_callback(self):
        if callable(self._value):
            return self._value

    def __call__(self, field, direction_symbol, index_field, **kwargs):

        if field.index_dimensions == 0:
            return [SympyAssignment(field.center, index_field("value") if self.additional_data else self._value)]
        elif field.index_dimensions == 1:
            assert not self.additional_data
            if not field.has_fixed_index_shape:
                raise NotImplementedError("Field needs fixed index shape")
            assert len(self._value) == field.index_shape[0], "Dirichlet value does not match index shape of field"
            return [SympyAssignment(field(i), self._value[i]) for i in range(field.index_shape[0])]
        raise NotImplementedError("Dirichlet boundary not implemented for fields with more than one index dimension")
