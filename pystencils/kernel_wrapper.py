import pystencils


class KernelWrapper:
    """
    Light-weight wrapper around a compiled kernel.

    Can be called while still providing access to underlying AST.
    """

    def __init__(self, kernel, parameters, ast_node: pystencils.astnodes.KernelFunction):
        self.kernel = kernel
        self.parameters = parameters
        self.ast = ast_node
        self.num_regs = None

    def __call__(self, **kwargs):
        return self.kernel(**kwargs)

    @property
    def code(self):
        return pystencils.get_code_str(self.ast)
