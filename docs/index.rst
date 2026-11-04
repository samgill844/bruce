BRUCE: exoplanet modelling and analysis
=======================================

BRUCE combines analytical transit and Keplerian radial-velocity models with
photometric data processing, transit-template searches and sampling utilities.
Its numerical core is written in C, with optional OpenMP parallelism, and is
adapted from Pierre Maxted's **ellc** (see the repository's NOTICE.md).

Start with installation, then run the examples before exploring the API.
The Python modelling examples use seeded synthetic data; the command-line
guides demonstrate real TESS data and follow-up planning.
The Python import is **bruce**, in lowercase.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   conventions
   examples/transit
   examples/radial_velocity
   examples/template_search
   examples/orbit

.. toctree::
   :maxdepth: 2
   :caption: Command-line tools

   cli/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/binarystar
   api/template_match
   api/visual
   api/data
   api/sampler
   api/bruce_c

.. toctree::
   :maxdepth: 1
   :caption: Documentation development

   building
   contributing

* :ref:`genindex`
* :ref:`search`
