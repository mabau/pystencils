from typing import List
from pystencils.astnodes import Node


# TODO ABC for NodeCollection and AssignmentCollection
class NodeCollection:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.bound_fields = None
        self.rhs_fields = None
        self.simplification_hints = ()

    @property
    def all_assignments(self):
        return self.nodes
