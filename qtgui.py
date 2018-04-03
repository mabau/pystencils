import sys
from PyQt5.QtWidgets import QWidget, QApplication, QTreeWidget, QTreeWidgetItem, QHBoxLayout
from pystencils.astnodes import Block, LoopOverCoordinate, KernelFunction


def debug_gui(ast):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
    ex = DebugTree()
    ex.insert_ast(ast)
    app.exec_()


class DebugTree(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(1)
        self.tree.setHeaderLabel('repr')

        hbox = QHBoxLayout()
        hbox.stretch(1)
        hbox.addWidget(self.tree)

        self.setWindowTitle('Debug')
        self.setLayout(hbox)
        self.show()

    def insert_ast(self, node, parent=None):
        if parent is None:
            parent = self.tree
        if isinstance(node, Block):  # Blocks are represented with the tree structure
            item = parent
        else:
            item = QTreeWidgetItem(parent)
            item.setText(0, repr(node))

        if node.func in [LoopOverCoordinate, KernelFunction]:
            self.tree.expandItem(item)

        for child in node.args:
            self.insert_ast(child, item)
