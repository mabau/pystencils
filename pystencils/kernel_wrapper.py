"""
Light-weight wrapper around a compiled kernel
"""
import pystencils


class KernelWrapper:
    def __init__(self, kernel, parameters, ast_node):
        self.kernel = kernel
        self.parameters = parameters
        self.ast = ast_node
        self.num_regs = None

    def __call__(self, **kwargs):
        return self.kernel(**kwargs)

    @property
    def code(self):
        return str(pystencils.show_code(self.ast))
