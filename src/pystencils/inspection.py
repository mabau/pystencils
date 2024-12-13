from typing import overload

from .backend.ast import PsAstNode
from .backend.emission import CAstPrinter, IRAstPrinter, EmissionError
from .backend.kernelfunction import KernelFunction
from .kernelcreation import StageResult, CodegenIntermediates
from abc import ABC, abstractmethod

_UNABLE_TO_DISPLAY_CPP = """
<div>
    <b>Unable to display C code for this abstract syntax tree</b>
    <p>
    This intermediate abstract syntax tree contains nodes that cannot be
    printed as valid C code.
    </p>
</div>
"""

_GRAPHVIZ_NOT_IMPLEMENTED = """
<div>
    <b>AST Visualization Unavailable</b>
    <p>
    AST visualization using GraphViz is not implemented yet.
    </p>
</div>
"""

_ERR_MSG = """
<div style="font-family: monospace; background-color: #EEEEEE; white-space: nowrap; overflow-x: scroll">
    {}
</div>
"""


class CodeInspectionBase(ABC):
    def __init__(self) -> None:
        self._ir_printer = IRAstPrinter(annotate_constants=False)
        self._c_printer = CAstPrinter()

    def _ir_tab(self, ir_obj: PsAstNode | KernelFunction):
        import ipywidgets as widgets

        ir = self._ir_printer(ir_obj)
        ir_tab = widgets.HTML(self._highlight_as_cpp(ir))
        self._apply_tab_layout(ir_tab)
        return ir_tab

    def _cpp_tab(self, ir_obj: PsAstNode | KernelFunction):
        import ipywidgets as widgets

        try:
            cpp = self._c_printer(ir_obj)
            cpp_tab = widgets.HTML(self._highlight_as_cpp(cpp))
        except EmissionError as e:
            cpp_tab = widgets.VBox(
                children=[
                    widgets.HTML(_UNABLE_TO_DISPLAY_CPP),
                    widgets.Accordion(
                        children=[widgets.HTML(_ERR_MSG.format(e.args[0]))],
                        titles=["Error Details"],
                    ),
                ]
            )
        self._apply_tab_layout(cpp_tab)
        return cpp_tab

    def _graphviz_tab(self, ir_obj: PsAstNode | KernelFunction):
        import ipywidgets as widgets

        graphviz_tab = widgets.HTML(_GRAPHVIZ_NOT_IMPLEMENTED)
        self._apply_tab_layout(graphviz_tab)
        return graphviz_tab

    def _apply_tab_layout(self, tab):
        tab.layout.display = "inline-block"
        tab.layout.padding = "0 15pt 0 0"

    def _highlight_as_cpp(self, code: str) -> str:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import CppLexer

        formatter = HtmlFormatter(
            prestyles="white-space: pre;",
        )
        html_code = highlight(code, CppLexer(), formatter)
        return html_code

    def _ipython_display_(self):
        from IPython.display import display

        display(self._widget())

    @abstractmethod
    def _widget(self): ...


class AstInspection(CodeInspectionBase):
    """Inspect an abstract syntax tree produced by the code generation backend.

    **Interactive:** This class can be used in Jupyter notebooks to interactively
    explore an abstract syntax tree.
    """

    def __init__(
        self,
        ast: PsAstNode,
        show_ir: bool = True,
        show_cpp: bool = True,
        show_graph: bool = True,
    ):
        super().__init__()
        self._ast = ast
        self._show_ir = show_ir
        self._show_cpp = show_cpp
        self._show_graph = show_graph

    def _widget(self):
        import ipywidgets as widgets

        tabs = []
        if self._show_ir:
            tabs.append(self._ir_tab(self._ast))
        if self._show_cpp:
            tabs.append(self._cpp_tab(self._ast))
        if self._show_graph:
            tabs.append(self._graphviz_tab(self._ast))

        tabs = widgets.Tab(children=tabs)
        tabs.titles = ["IR Code", "C Code", "AST Visualization"]

        tabs.layout.height = "250pt"

        return tabs


class KernelInspection(CodeInspectionBase):
    def __init__(
        self,
        kernel: KernelFunction,
        show_ir: bool = True,
        show_cpp: bool = True,
        show_graph: bool = True,
    ) -> None:
        super().__init__()
        self._kernel = kernel
        self._show_ir = show_ir
        self._show_cpp = show_cpp
        self._show_graph = show_graph

    def _widget(self):
        import ipywidgets as widgets

        tabs = []
        if self._show_ir:
            tabs.append(self._ir_tab(self._kernel))
        if self._show_cpp:
            tabs.append(self._cpp_tab(self._kernel))
        if self._show_graph:
            tabs.append(self._graphviz_tab(self._kernel))

        tabs = widgets.Tab(children=tabs)
        tabs.titles = ["IR Code", "C Code", "AST Visualization"]

        tabs.layout.height = "250pt"

        return tabs


class IntermediatesInspection:
    def __init__(
        self,
        intermediates: CodegenIntermediates,
        show_ir: bool = True,
        show_cpp: bool = True,
        show_graph: bool = True,
    ):
        self._intermediates = intermediates
        self._show_ir = show_ir
        self._show_cpp = show_cpp
        self._show_graph = show_graph

    def _ipython_display_(self):
        from IPython.display import display
        import ipywidgets as widgets

        stages = self._intermediates.available_stages

        previews: list[AstInspection] = [
            AstInspection(
                stage.ast,
                show_ir=self._show_ir,
                show_cpp=self._show_cpp,
                show_graph=self._show_graph,
            )
            for stage in stages
        ]
        labels: list[str] = [stage.label for stage in stages]

        code_views = [p._widget() for p in previews]
        for v in code_views:
            v.layout.width = "100%"

        select_label = widgets.HTML("<div><b>Code Generator Stages</b></div>")
        select = widgets.Select(options=labels)
        select.layout.height = "250pt"

        selection_box = widgets.VBox([select_label, select])
        selection_box.layout.overflow = "visible"

        preview_label = widgets.HTML("<div><b>Preview</b></div>")
        preview_stack = widgets.Stack(children=code_views)
        preview_stack.layout.overflow = "hidden"

        preview_box = widgets.VBox([preview_label, preview_stack])

        widgets.jslink((select, "index"), (preview_stack, "selected_index"))

        grid = widgets.GridBox(
            [selection_box, preview_box],
            layout=widgets.Layout(grid_template_columns="max-content auto"),
        )

        display(grid)


@overload
def inspect(obj: PsAstNode): ...


@overload
def inspect(obj: KernelFunction): ...


@overload
def inspect(obj: StageResult): ...


@overload
def inspect(obj: CodegenIntermediates): ...


def inspect(obj, show_ir: bool = True, show_cpp: bool = True, show_graph: bool = True):
    """Interactively inspect various products of the code generator.

    When run inside a Jupyter notebook, this function displays an inspection widget
    for the following types of objects:
    - `PsAstNode`
    - `KernelFunction`
    - `StageResult`
    - `CodegenIntermediates`
    """

    from IPython.display import display

    match obj:
        case PsAstNode():
            preview = AstInspection(
                obj, show_ir=show_ir, show_cpp=show_cpp, show_graph=show_cpp
            )
        case KernelFunction():
            preview = KernelInspection(
                obj, show_ir=show_ir, show_cpp=show_cpp, show_graph=show_cpp
            )
        case StageResult(ast, _):
            preview = AstInspection(
                ast, show_ir=show_ir, show_cpp=show_cpp, show_graph=show_cpp
            )
        case CodegenIntermediates():
            preview = IntermediatesInspection(
                obj, show_ir=show_ir, show_cpp=show_cpp, show_graph=show_cpp
            )
        case _:
            raise ValueError(f"Cannot inspect object of type {type(obj)}")

    display(preview)
