from collections import defaultdict

import sympy as sp

from pystencils.field import Field
from pystencils.sympyextensions import multidimensional_sum, prod
from pystencils.utils import LinearEquationSystem, fully_contains


class FiniteDifferenceStencilDerivation:
    """Derives finite difference stencils.

    Can derive standard finite difference stencils, as well as isotropic versions
    (see Isotropic Finite Differences by A. Kumar)

    Args:
        derivative_coordinates: tuple indicating which derivative should be approximated,
                                (1, ) stands for first derivative in second direction (y),
                                (0, 1) would be a mixed second derivative in x and y
                                (0, 0, 0) would be a third derivative in x direction
        stencil: list of offset tuples, defining the stencil
        dx: spacing between grid points, one for all directions, i.e. dx=dy=dz

    Examples:
        Central differences
        >>> fd_1d = FiniteDifferenceStencilDerivation((0,), stencil=[(-1,), (0,), (1,)])
        >>> result = fd_1d.get_stencil()
        >>> result
        Finite difference stencil of accuracy 2, isotropic error: False
        >>> result.weights
        [-1/2, 0, 1/2]

        Forward differences
        >>> fd_1d = FiniteDifferenceStencilDerivation((0,), stencil=[(0,), (1,)])
        >>> result = fd_1d.get_stencil()
        >>> result
        Finite difference stencil of accuracy 1, isotropic error: False
        >>> result.weights
        [-1, 1]
    """

    def __init__(self, derivative_coordinates, stencil, dx=1):
        self.dim = len(stencil[0])
        self.field = Field.create_generic('f', spatial_dimensions=self.dim)
        self._derivative = tuple(sorted(derivative_coordinates))
        self._stencil = stencil
        self._dx = dx
        self.weights = {tuple(d): self.symbolic_weight(*d) for d in self._stencil}

    def assume_symmetric(self, dim, anti_symmetric=False):
        """Adds restriction that weight in opposite directions of a dimension are equal (symmetric) or
        the negative of each other (anti symmetric)

        For example: dim=1, assumes that w(1, 1) == w(1, -1), if anti_symmetric=False or
                                         w(1, 1) == -w(1, -1) if anti_symmetric=True
        """
        update = {}
        for direction, value in self.weights.items():
            inv_direction = tuple(-offset if i == dim else offset for i, offset in enumerate(direction))
            if direction[dim] < 0:
                inv_weight = self.weights[inv_direction]
                update[direction] = -inv_weight if anti_symmetric else inv_weight
        self.weights.update(update)

    def set_weight(self, offset, value):
        assert offset in self.weights
        self.weights[offset] = value

    def get_stencil(self, isotropic=False) -> 'FiniteDifferenceStencilDerivation.Result':
        weights = [self.weights[d] for d in self._stencil]
        system = LinearEquationSystem(sp.Matrix(weights).atoms(sp.Symbol))

        order = 0

        while True:
            new_system = system.copy()
            eq = self.error_term_equations(order)
            new_system.add_equations(eq)
            sol_structure = new_system.solution_structure()
            if sol_structure == 'single':
                system = new_system
            elif sol_structure == 'multiple':
                system = new_system
            elif sol_structure == 'none':
                break
            else:
                assert False
            order += 1

        accuracy = order - len(self._derivative)
        error_is_isotropic = False
        if isotropic:
            new_system = system.copy()
            new_system.add_equations(self.isotropy_equations(order))
            sol_structure = new_system.solution_structure()
            error_is_isotropic = sol_structure != 'none'
            if error_is_isotropic:
                system = new_system

        solve_res = system.solution()
        weight_list = [self.weights[d].subs(solve_res) for d in self._stencil]
        return self.Result(self._stencil, weight_list, accuracy, error_is_isotropic)

    @staticmethod
    def symbolic_weight(*args):
        str_args = [str(e) for e in args]
        return sp.Symbol("w_({})".format(",".join(str_args)))

    def error_term_dict(self, order):
        error_terms = defaultdict(lambda: 0)
        for direction in self._stencil:
            weight = self.weights[tuple(direction)]
            x = tuple(self._dx * d_i for d_i in direction)
            for offset in multidimensional_sum(order, dim=self.field.spatial_dimensions):
                fac = sp.factorial(order)
                error_terms[tuple(sorted(offset))] += weight / fac * prod(x[off] for off in offset)
        if self._derivative in error_terms:
            error_terms[self._derivative] -= 1
        return error_terms

    def error_term_equations(self, order):
        return list(self.error_term_dict(order).values())

    def isotropy_equations(self, order):
        def cycle_int_sequence(sequence, modulus):
            import numpy as np
            result = []
            arr = np.array(sequence, dtype=int)
            while True:
                if tuple(arr) in result:
                    break
                result.append(tuple(arr))
                arr = (arr + 1) % modulus
            return tuple(set(tuple(sorted(t)) for t in result))

        error_dict = self.error_term_dict(order)
        eqs = []
        for derivative_tuple in list(error_dict.keys()):
            if fully_contains(self._derivative, derivative_tuple):
                remaining = list(derivative_tuple)
                for e in self._derivative:
                    del remaining[remaining.index(e)]
                permutations = cycle_int_sequence(remaining, self.dim)
                if len(permutations) == 1:
                    eqs.append(error_dict[derivative_tuple])
                else:
                    for i in range(1, len(permutations)):
                        new_eq = (error_dict[tuple(sorted(permutations[i] + self._derivative))]
                                  - error_dict[tuple(sorted(permutations[i - 1] + self._derivative))])
                        if new_eq:
                            eqs.append(new_eq)
            else:
                eqs.append(error_dict[derivative_tuple])
        return eqs

    class Result:
        def __init__(self, stencil, weights, accuracy, is_isotropic):
            self.stencil = stencil
            self.weights = weights
            self.accuracy = accuracy
            self.is_isotropic = is_isotropic

        def visualize(self):
            from pystencils.stencil import plot
            plot(self.stencil, data=self.weights)

        def apply(self, field_access: Field.Access):
            f = field_access
            return sum(f.get_shifted(*offset) * weight for offset, weight in zip(self.stencil, self.weights))

        def as_matrix(self):
            dim = len(self.stencil[0])
            assert dim == 2
            max_offset = max(max(abs(e) for e in direction) for direction in self.stencil)
            result = sp.Matrix(2 * max_offset + 1, 2 * max_offset + 1, lambda i, j: 0)
            for direction, weight in zip(self.stencil, self.weights):
                result[max_offset - direction[1], max_offset + direction[0]] = weight
            return result

        def __repr__(self):
            return "Finite difference stencil of accuracy {}, isotropic error: {}".format(self.accuracy,
                                                                                          self.is_isotropic)
