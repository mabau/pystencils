***************************************
Assignment Collection & Simplifications
***************************************


AssignmentCollection
====================

.. autoclass:: pystencils.AssignmentCollection
   :members:


SimplificationStrategy
======================

.. autoclass:: pystencils.simp.SimplificationStrategy
    :members:

Simplifications
===============

.. automodule:: pystencils.simp.simplifications
    :members:

Subexpression insertion
=======================

The subexpression insertions have the goal to insert subexpressions which will not reduce the number of FLOPs.
For example a constant value kept as subexpression will lead to a new variable in the code which will occupy
a register slot. On the other side a single variable could just be inserted in all assignments.

.. automodule:: pystencils.simp.subexpression_insertion
    :members:





