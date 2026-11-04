# Contributing and getting help

BRUCE is maintained by Samuel Gill. Suggestions, reproducible bug reports,
documentation improvements and pull requests are welcome. The maintainer reviews
proposed changes and decides what enters a release; response times depend on
research commitments.

Use [GitHub Issues](https://github.com/samgill844/bruce/issues) for support,
questions and bug reports. Include your Python/BRUCE versions, operating system,
installation command, complete traceback and a small reproducible example.
For numerical problems, include parameter values, units, expected behaviour and
an independent reference where possible. Use synthetic or public data; do not
attach credentials or large archive downloads.

Discuss substantial changes in an issue before implementing them. Create a
branch, keep a pull request focused, explain the scientific/technical reason,
and update relevant examples and argument documentation. Preserve existing
public names and parameter conventions unless a breaking change is discussed.
Contributions are distributed under BRUCE's GPL-3.0-or-later licence; preserve
upstream attribution in LICENSE and NOTICE.md.

From a checkout, install development and validation dependencies:

```bash
python -m pip install -e '.[analysis,gui]'
python -m pip install -r docs/requirements.txt -r paper/requirements-paper.txt
OMP_NUM_THREADS=1 python -m unittest discover -s tests/regression -v
python docs/scripts/generate_api.py
python docs/scripts/generate_c_api.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
python paper/scripts/check_paper.py
```

The ellc comparison is optional and is skipped when ellc is unavailable.
The paper preview additionally needs the external wkhtmltopdf executable.
The CI workflow exercises core regression tests and examples on Linux; GUI
windows and network-dependent archive retrieval need manual checks. For GUI
changes, run the relevant command from docs/cli/ and describe what you checked.
For numerical changes, include meaningful validation of accuracy, units and
limiting cases; repeat performance measurements if making new timing claims.

Please keep discussions respectful and give enough detail for others to
reproduce and assess a proposed change.
