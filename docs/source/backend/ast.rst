********************
Abstract Syntax Tree
********************

.. automodule:: pystencils.backend.ast

API Documentation
=================

Inheritance Diagram
-------------------

.. inheritance-diagram:: pystencils.backend.ast.astnode.PsAstNode pystencils.backend.ast.structural pystencils.backend.ast.expressions pystencils.backend.ast.vector pystencils.backend.extensions.foreign_ast
    :top-classes: pystencils.types.PsAstNode
    :parts: 1

Base Classes
------------

.. module:: pystencils.backend.ast.astnode

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsAstNode
    PsLeafMixIn


Structural Nodes
----------------

.. module:: pystencils.backend.ast.structural

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsBlock
    PsStatement
    PsAssignment
    PsDeclaration
    PsLoop
    PsConditional
    PsEmptyLeafMixIn
    PsPragma
    PsComment


Expressions
-----------

.. module:: pystencils.backend.ast.expressions

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsExpression
    PsLvalue
    PsSymbolExpr
    PsConstantExpr
    PsLiteralExpr
    PsBufferAcc
    PsSubscript
    PsMemAcc
    PsLookup
    PsCall
    PsTernary
    PsNumericOpTrait
    PsIntOpTrait
    PsBoolOpTrait
    PsUnOp
    PsNeg
    PsAddressOf
    PsCast
    PsBinOp
    PsAdd
    PsSub
    PsMul
    PsDiv
    PsIntDiv
    PsRem
    PsLeftShift
    PsRightShift
    PsBitwiseAnd
    PsBitwiseXor
    PsBitwiseOr
    PsAnd
    PsOr
    PsNot
    PsRel
    PsEq
    PsNe
    PsGe
    PsLe
    PsGt
    PsLt
    PsArrayInitList


SIMD Nodes
----------

.. module:: pystencils.backend.ast.vector

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsVectorOp
    PsVecBroadcast
    PsVecMemAcc


Utility
-------

.. currentmodule:: pystencils.backend.ast

.. autosummary::
    :toctree: generated
    :nosignatures:

    expressions.evaluate_expression
    dfs_preorder
    dfs_postorder
    util.AstEqWrapper
    util.determine_memory_object
