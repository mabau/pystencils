import sympy as sp

from pystencils import AssignmentCollection, Assignment
from pystencils.node_collection import NodeCollection
from pystencils.astnodes import SympyAssignment


def test_node_collection_from_assignment_collection():
    x = sp.symbols('x')
    assignment_collection = AssignmentCollection([Assignment(x, 2)])
    node_collection = NodeCollection.from_assignment_collection(assignment_collection)

    assert node_collection.all_assignments[0] == SympyAssignment(x, 2)
