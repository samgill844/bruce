Command-line tools
==================

BRUCE installs three interactive tools for inspecting TESS observations and
planning follow-up. These examples use TIC 100504488 and the screenshots
supplied by Samuel Gill. They complement the synthetic Python examples.

Install the analysis and GUI dependencies together:

.. code-block:: bash

   python -m pip install 'bruce[analysis,gui] @ git+https://github.com/samgill844/bruce.git'

``spocfit`` and ``tesstpf`` open Qt windows and need a graphical desktop.
The tools retrieve target information and TESS products from MAST, so allow
network access and time for the first download. ``predict`` needs the analysis
extra but does not require Qt. Commands are installed in the active Python
environment's executable directory.

.. toctree::
   :maxdepth: 1

   spocfit
   tesstpf
   predict
