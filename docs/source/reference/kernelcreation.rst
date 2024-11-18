.. _page_kernel_creation:

***************
Kernel Creation
***************

Targets
=======

.. module:: pystencils.target

.. autosummary::
    :toctree: autoapi
    :nosignatures:
    :template: autosummary/recursive_class.rst

    Target


Configuration
=============

.. module:: pystencils.config

.. autosummary::
    :toctree: autoapi
    :nosignatures:
    :template: autosummary/entire_class.rst

    CreateKernelConfig
    CpuOptimConfig
    OpenMpConfig
    VectorizationConfig
    GpuIndexingConfig


Creation
========

.. module:: pystencils.kernelcreation

.. autosummary::
    :toctree: autoapi
    :nosignatures:

    create_kernel


Kernel Parameters and Function Objects
======================================

.. module:: pystencils.backend.kernelfunction

.. autosummary::
    :toctree: autoapi
    :nosignatures:
    :template: autosummary/entire_class.rst

    KernelParameter
    KernelFunction
    GpuKernelFunction
