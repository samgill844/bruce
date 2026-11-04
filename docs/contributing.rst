Contributing, support and verification
======================================

Use the `GitHub issue tracker <https://github.com/samgill844/bruce/issues>`_
to report problems or ask for help. Include your operating system, Python and
BRUCE versions, the full error message and a minimal reproducing example.
For modelling problems, include units and parameter values. Synthetic or public
data make an issue easier for others to reproduce.

Documentation improvements and focused pull requests are welcome. Discuss
substantial changes in an issue first. Samuel Gill maintains BRUCE and reviews
proposed changes for inclusion; support is provided as research commitments
allow. See the repository's
`CONTRIBUTING.md <https://github.com/samgill844/bruce/blob/main/CONTRIBUTING.md>`_
for the development workflow and licence expectations.

Verify a source checkout
------------------------

With BRUCE's analysis dependencies installed, run from the repository root:

.. code-block:: bash

   OMP_NUM_THREADS=1 python -m unittest discover -s tests/regression -v

This checks model likelihoods, exposure handling, data utilities and the
Kepler solver. The independent ellc comparison runs if ellc is installed and
is otherwise skipped. The small examples under ``docs/examples/`` provide
further reproducible checks; see :doc:`building` for their commands.
GUI use and archive downloads need a graphical environment and network access
and are not exercised by the automatic core checks.

The repository includes a Linux test/documentation workflow and a separate
JOSS paper compilation workflow. Inspect their results in GitHub Actions for
the commit being reviewed. A workflow definition alone is not evidence of a
successful remote run.
