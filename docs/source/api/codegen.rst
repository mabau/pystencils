pystencils.codegen
==================

.. module:: pystencils.codegen

Invocation
----------

.. autosummary::
  :toctree: generated
  :nosignatures:

  create_kernel
  
Configuration
-------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CreateKernelConfig
  CpuOptimConfig
  OpenMpConfig
  VectorizationConfig
  GpuIndexingConfig

.. autosummary::
  :toctree: generated
  :nosignatures:

  AUTO

Target Specification
--------------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/recursive_class.rst

  Target

Code Generation Drivers
-----------------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  driver.DefaultKernelCreationDriver

.. autosummary::
  :toctree: generated
  :nosignatures:

  get_driver

Output Code Objects
-------------------

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  Kernel
  GpuKernel
  Parameter
  GpuThreadsRange
