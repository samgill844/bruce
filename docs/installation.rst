Installation
============

Install from GitHub
-------------------

Use Python 3.9 or later in a virtual environment. BRUCE compiles a C extension,
so Git and a C compiler are needed. On macOS install Apple's Command Line Tools
(``xcode-select --install``); on Linux install your distribution's C build tools.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install 'bruce[analysis] @ git+https://github.com/samgill844/bruce.git'

The analysis extra supplies plotting, sampling and archive dependencies used by
these tutorials. For numerical modelling alone, omit ``[analysis]``.
This installs the current default branch; once a release tag exists, append
``@v1.0.0`` to the Git URL to pin that release. Do not use ``pip install bruce``:
the PyPI project with that name is unrelated.

For the interactive ``spocfit`` and ``tesstpf`` programs, install both extras:

.. code-block:: bash

   python -m pip install 'bruce[analysis,gui] @ git+https://github.com/samgill844/bruce.git'

See :doc:`cli/index` for worked command-line examples.

OpenMP
------

On macOS, optional parallelism needs LLVM's OpenMP runtime:

.. code-block:: bash

   brew install libomp

The build searches standard Homebrew locations. For a custom installation set
``BRUCE_LIBOMP_PREFIX`` to its prefix. GCC builds normally enable OpenMP.
To explicitly build without it:

.. code-block:: bash

   BRUCE_OPENMP=0 python -m pip install 'bruce[analysis] @ git+https://github.com/samgill844/bruce.git'

Set ``OMP_NUM_THREADS`` **before starting Python**, for example
``OMP_NUM_THREADS=4 python my_analysis.py``. More threads can add overhead for
small datasets; avoid using a large thread count inside a multiprocessing fit.

Check the installation
----------------------

.. code-block:: python

   import numpy as np
   from bruce.binarystar import lc

   model = lc(np.linspace(-0.1, 0.1, 101))
   print(model.shape, model.min())

The model should contain 101 finite flux values and a dip below unity.
If the compiler cannot find NumPy headers, use the GitHub command above with
build isolation enabled (the default). If ``import bruce_c`` fails, check that
pip and Python belong to the same environment.
