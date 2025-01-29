#################################
pystencils v2.0-dev Documentation
#################################

.. note::
   You are currently viewing the documentation pages for the development revision |release|
   of pystencils 2.0.
   These pages have been generated from the branch 
   `v2.0-dev <https://i10git.cs.fau.de/pycodegen/pystencils/-/tree/v2.0-dev?ref_type=heads>`_.

   Pystencils 2.0 is currently under development. 
   It marks a complete re-design of the package's internal structure;
   furthermore, it will bring a set of new features and API changes.
   Be aware that many features are still missing or might have brand-new bugs in this version.
   If you wish to work with and contribute to this development, please refer to
   `the Git repository <https://i10git.cs.fau.de/pycodegen/pystencils/-/tree/v2.0-dev?ref_type=heads>`_.

.. note::
   These pages are still under construction; many aspects are still missing.
   Do not hesitate to contribute!

Welcome to the documentation and reference guide of *pystencils*!
*Pystencils* offers a symbolic language and code generator for the development of high-performing
numerical kernels for both CPU and GPU targets. 
Its features include:

- **Symbolic Algebra:** Design numerical methods on a mathematical level using the full power
  of the `SymPy <https://sympy.org>`_ computer algebra system.
  Make use of pystencils' discretization engines to automatically derive finite difference- and finite volume-methods,
  and take control of numerical precision using the `versatile type system <page_type_system>`.
- **Kernel Description:** Derive and optimize stencil-based update rules using a symbolic abstraction
  of numerical `fields <page_symbolic_language>`.
- **Code Generation:** `Generate and compile <guide_kernelcreation>` high-performance parallel kernels for CPUs and GPUs.
  Accelerate your kernels on multicore CPUs using the automatic OpenMP parallelization
  and make full use of your cores' SIMD units through the highly configurable vectorizer.
- **Rapid Prototyping:** Run your numerical solvers on `NumPy <https://numpy.org>`_ and `CuPy <https://cupy.dev>`_ arrays
  and test them interactively inside `Jupyter <https://jupyter.org/>`_ notebooks.
  Quickly set up numerical schemes, apply initial and boundary conditions, evaluate them on model problems
  and rapidly visualize the results using matplotlib or VTK.
- **Framework Integration:** Export your kernels and use them inside HPC frameworks
  such as `waLBerla`_ to build massively parallel simulations.


.. .. card:: Getting Started: Our Tutorials
..    :link: page_tutorials
..    :link-type: ref

..    New to *pystencils*? Check out our set of tutorials to quickly and interactively learn the basics.

.. .. card:: Reference Guide and APIs
..    :link: page_api
..    :link-type: ref

..    Get an overview of *pystencils*' APIs for mathematical modelling and code generation.

.. .. card:: Migration Guide: 1.3.x to 2.0
..    :link: page_v2_migration
..    :link-type: ref

..    Find advice on migrating your code from *pystencils 1.3.x* to *pystencils 2.0*

.. .. card:: Developers's Reference: Code Generation Backend
..    :link: page_codegen_backend
..    :link-type: ref

..    Dive deep into the core of pystencils' code generation engine.

Topics
------

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  installation
  tutorials/index

.. toctree::
  :maxdepth: 1
  :caption: User Manual

  user_manual/symbolic_language
  user_manual/kernelcreation
  user_manual/gpu_kernels
  user_manual/WorkingWithTypes

.. toctree::
  :maxdepth: 1
  :caption: API Reference

  api/symbolic/index
  api/types
  api/codegen
  api/jit

.. toctree::
  :maxdepth: 1
  :caption: Topics

  contributing/index
  migration
  backend/index

Projects using pystencils
-------------------------

- `lbmpy <https://pycodegen.pages.i10git.cs.fau.de/lbmpy/>`_
- `waLBerla`_
- `HyTeG Operator Generator (HOG) <https://hyteg.pages.i10git.cs.fau.de/hog/>`_


.. _walberla: https://walberla.net
