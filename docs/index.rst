BRUCE: exoplanet modelling and analysis
=======================================

BRUCE combines analytical transit and Keplerian radial-velocity models with
photometric data processing, transit-template searches and sampling utilities.
Its numerical core is written in C, with optional OpenMP parallelism, and is
adapted from Pierre Maxted's **ellc** (see the repository's NOTICE.md).

Start with installation, then run the examples before exploring the API.
All examples use seeded synthetic data; no archive access is required.
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
   :caption: API reference

   api/binarystar
   api/template_match
   api/visual
   api/data
   api/sampler

.. toctree::
   :maxdepth: 1
   :caption: Documentation development

   building

* :ref:`genindex`
* :ref:`search`
