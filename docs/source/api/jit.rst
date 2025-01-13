pystencils.jit
==============

.. module:: pystencils.jit

Base Infrastructure
-------------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

    KernelWrapper
    JitBase
    NoJit

.. autodata:: no_jit

Legacy CPU JIT
--------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  LegacyCpuJit

CuPy-based GPU JIT
------------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CupyJit
  CupyKernelWrapper
  LaunchGrid
