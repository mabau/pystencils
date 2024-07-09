import itertools
from collections import defaultdict

import numpy as np
import sympy as sp

from pystencils.field import Field
from pystencils.stencil import direction_string_to_offset
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
        return sp.Symbol(f"w_({','.join(str_args)})")

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

        def __array__(self):
            return np.array(self.as_array().tolist())

        def as_array(self):
            dim = len(self.stencil[0])
            assert (dim == 2 or dim == 3), "Only 2D or 3D matrix representations are available"
            max_offset = max(max(abs(e) for e in direction) for direction in self.stencil)
            shape_list = []
            for i in range(dim):
                shape_list.append(2 * max_offset + 1)

            number_of_elements = np.prod(shape_list)
            shape = tuple(shape_list)
            result = sp.MutableDenseNDimArray([0] * number_of_elements, shape)

            if dim == 2:
                for direction, weight in zip(self.stencil, self.weights):
                    result[max_offset - direction[1], max_offset + direction[0]] = weight
            if dim == 3:
                for direction, weight in zip(self.stencil, self.weights):
                    result[max_offset - direction[1], max_offset + direction[0], max_offset + direction[2]] = weight

            return result

        def rotate_weights_and_apply(self, field_access: Field.Access, axes):
            """derive gradient weights of other direction with already calculated weights of one direction
               via rotation and apply them to a field."""
            dim = len(self.stencil[0])
            assert (dim == 2 or dim == 3), "This function is only for 2D or 3D stencils available"
            rotated_weights = np.rot90(np.array(self.__array__()), 1, axes)

            result = []
            max_offset = max(max(abs(e) for e in direction) for direction in self.stencil)
            if dim == 2:
                for direction in self.stencil:
                    result.append(rotated_weights[max_offset - direction[1],
                                                  max_offset + direction[0]])
            if dim == 3:
                for direction in self.stencil:
                    result.append(rotated_weights[max_offset - direction[1],
                                                  max_offset + direction[0],
                                                  max_offset + direction[2]])

            f = field_access
            return sum(f.get_shifted(*offset) * weight for offset, weight in zip(self.stencil, result))

        def __repr__(self):
            return "Finite difference stencil of accuracy {}, isotropic error: {}".format(self.accuracy,
                                                                                          self.is_isotropic)


class FiniteDifferenceStaggeredStencilDerivation:
    """Derives a finite difference stencil for application at a staggered position

    Args:
        neighbor: the neighbor direction string or vector at whose staggered position to calculate the derivative
        dim: how many dimensions (2 or 3)
        derivative: a tuple of directions over which to perform derivatives
        free_weights_prefix: a string to prefix to free weight symbols. If None, do not return free weights
    """

    def __init__(self, neighbor, dim, derivative=tuple(), free_weights_prefix=None):
        if type(neighbor) is str:
            neighbor = direction_string_to_offset(neighbor)
        if dim == 2:
            assert neighbor[dim:] == 0
        assert derivative is tuple() or max(derivative) < dim
        neighbor = sp.Matrix(neighbor[:dim])
        pos = neighbor / 2

        def unitvec(i):
            """return the `i`-th unit vector in three dimensions"""
            a = np.zeros(dim, dtype=int)
            a[i] = 1
            return a

        def flipped(a, i):
            """return `a` with its `i`-th element's sign flipped"""
            a = a.copy()
            a[i] *= -1
            return a

        # determine the points to use, coordinates are relative to position
        points = []
        if np.linalg.norm(neighbor, 1) == 1:
            main_points = [neighbor / 2, neighbor / -2]
        elif np.linalg.norm(neighbor, 1) == 2:
            nonzero_indices = [i for i, v in enumerate(neighbor) if v != 0 and i < dim]
            main_points = [neighbor / 2, neighbor / -2, flipped(neighbor / 2, nonzero_indices[0]),
                           flipped(neighbor / -2, nonzero_indices[0])]
        else:
            main_points = [sp.Matrix(np.multiply(neighbor, sp.Matrix(c) / 2))
                           for c in itertools.product([-1, 1], repeat=3)]
        points += main_points
        zero_indices = [i for i, v in enumerate(neighbor) if v == 0 and i < dim]
        for i in zero_indices:
            points += [point + sp.Matrix(unitvec(i)) for point in main_points]
            points += [point - sp.Matrix(unitvec(i)) for point in main_points]
        points_tuple = tuple([tuple(p) for p in points])
        self._stencil = points_tuple

        # determine the stencil weights
        if len(derivative) == 0:
            weights = None
        else:
            derivation = FiniteDifferenceStencilDerivation(derivative, points_tuple).get_stencil()
            if not derivation.accuracy:
                raise Exception('the requested derivative cannot be performed with the available neighbors')
            weights = derivation.weights

            # if the weights are underdefined, we can choose the free symbols to find the sparsest stencil
            free_weights = set(itertools.chain(*[w.free_symbols for w in weights]))
            if free_weights_prefix is not None:
                weights = [w.subs({fw: sp.Symbol(f"{free_weights_prefix}_{i}") for i, fw in enumerate(free_weights)})
                           for w in weights]
            elif len(free_weights) > 0:
                zero_counts = defaultdict(list)
                for values in itertools.product([-1, -sp.Rational(1, 2), 0, 1, sp.Rational(1, 2)],
                                                repeat=len(free_weights)):
                    subs = {free_weight: value for free_weight, value in zip(free_weights, values)}
                    weights = [w.subs(subs) for w in derivation.weights]
                    if not all(a == 0 for a in weights):
                        zero_count = sum([1 for w in weights if w == 0])
                        zero_counts[zero_count].append(weights)
                best = zero_counts[max(zero_counts.keys())]
                if len(best) > 1:  # if there are multiple, pick the one that contains a nonzero center weight
                    center = [tuple(p + pos) for p in points].index((0, 0, 0)[:dim])
                    best = [b for b in best if b[center] != 0]
                if len(best) > 1:  # if there are still multiple, they are equivalent, so we average
                    weights = [sum([b[i] for b in best]) / len(best) for i in range(len(weights))]
                else:
                    weights = best[0]
                assert weights

        points_tuple = tuple([tuple(p + pos) for p in points])
        self._points = points_tuple
        self._weights = weights

    @property
    def points(self):
        """return the points of the stencil"""
        return self._points

    @property
    def stencil(self):
        """return the points of the stencil relative to the staggered position specified by neighbor"""
        return self._stencil

    @property
    def weights(self):
        """return the weights of the stencil"""
        assert self._weights is not None
        return self._weights

    def visualize(self):
        if self._weights is None:
            ws = None
        else:
            ws = np.array([w for w in self.weights if w != 0], dtype=float)
        pts = np.array([p for i, p in enumerate(self.points) if self.weights[i] != 0], dtype=int)
        from pystencils.stencil import plot
        plot(pts, data=ws)

    def apply(self, access: Field.Access):
        return sum([access.get_shifted(*point) * weight for point, weight in zip(self.points, self.weights)])
