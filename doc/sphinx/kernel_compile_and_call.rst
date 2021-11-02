*****************************************
Creating and calling kernels from Python
*****************************************


Creating kernels
----------------

.. autofunction:: pystencils.create_kernel

.. autoclass:: pystencils.CreateKernelConfig
    :members:

.. autofunction:: pystencils.create_domain_kernel

.. autofunction:: pystencils.create_indexed_kernel

.. autofunction:: pystencils.create_staggered_kernel


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
