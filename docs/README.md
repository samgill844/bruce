# BRUCE documentation

From the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open http://localhost:8000, or open `_build/html/index.html` directly.
The HTML build uses the committed examples, plots and GIF and does not require
BRUCE to be imported. To run the numerical examples, install BRUCE with its
analysis extra first; see `installation.rst`. See `building.rst` for regeneration
commands. Edit API explanations in `scripts/generate_api.py`, then rerun that
script so signatures stay consistent with the Python source.
