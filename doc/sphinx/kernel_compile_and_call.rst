*****************************************
Creating and calling kernels from Python
*****************************************


Creating kernels
----------------

.. autofunction:: pystencils.create_kernel

.. autoclass:: pystencils.CreateKernelConfig
    :members:

.. autofunction:: pystencils.kernelcreation.create_domain_kernel

.. autofunction:: pystencils.kernelcreation.create_indexed_kernel

.. autofunction:: pystencils.kernelcreation.create_staggered_kernel


Code printing
-------------

.. autofunction:: pystencils.show_code


GPU Indexing
-------------

.. autoclass:: pystencils.gpucuda.AbstractIndexing
   :members:

.. autoclass:: pystencils.gpucuda.BlockIndexing
   :members:

.. autoclass:: pystencils.gpucuda.LineIndexing
   :members:
