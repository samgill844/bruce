Build and view these docs
=========================

From the repository root, with BRUCE installed:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python docs/scripts/generate_api.py
   python docs/scripts/generate_c_api.py
   MPLBACKEND=Agg python docs/examples/transit.py
   MPLBACKEND=Agg python docs/examples/radial_velocity.py
   MPLBACKEND=Agg python docs/examples/template_search.py
   MPLBACKEND=Agg python docs/examples/orbit.py
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m http.server 8000 --directory docs/_build/html

Visit http://localhost:8000 in your browser. Stop the server with Ctrl-C.
You can also open ``docs/_build/html/index.html`` directly. Generated example
images are committed so normal documentation builds do not rerun the numerical
examples. Regenerate them after changing an example.

The API generator reads function signatures from the Python source using the
standard-library AST; it does not import the C extension, contact archives or
load GUI modules. It combines these signatures with the maintained descriptions
in ``docs/scripts/generate_api.py``. After adding an API function, add its
argument descriptions there and rerun the generator. The ``bruce_c`` reference
has its own generator, ``docs/scripts/generate_c_api.py``, which checks the
export list and positional argument order against the C wrappers. Array-valued defaults
are displayed as ``...``; their expressions are linked through the source files.

The generated HTML is local and ignored by Git. Hosting on GitHub Pages or
Read the Docs can be added later; this build does not publish anything.

Command-line screenshots live in ``docs/_static/cli/``. The supplied
``predict`` output lives in ``docs/cli/predict-output.txt``; neither archive
downloads nor GUI applications are run during the documentation build.
