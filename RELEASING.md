# Releasing BRUCE for JOSS

The package version is currently 1.0.0 in setup.py. A GitHub release is sufficient;
a PyPI upload is optional. The PyPI name bruce belongs to an unrelated project.

Before tagging:

1. Commit and push the final code, documentation, screenshots, paper and figures.
   Confirm LICENSE, NOTICE.md and CONTRIBUTING.md are included. Do not accidentally
   include local FITS downloads, HDF5 chains, build directories or credentials.
2. Check the **Tests and documentation** workflow on that commit. Run the optional
   ellc comparison locally as described in CONTRIBUTING.md.
3. Run the **JOSS paper** workflow in GitHub Actions and inspect its PDF artifact.
   The local paper/output/paper-preview.pdf is a reading copy; it does not verify
   compilation with the journal's toolchain. This workflow has been prepared
   locally but has not yet run remotely.
4. Check the documented spocfit and tesstpf GUI commands on your machine, and ask
   a colleague to try installation and a worked example in a fresh environment.
5. Review the author-specific items in paper/SUBMISSION_CHECKLIST.md.

Create a GitHub release targeting the verified commit with tag **v1.0.0** if that
version has not already been released. Set the release title to BRUCE 1.0.0 and
summarise the public functions, GUI tools, documentation and validation. Note the
corrected eccentric epoch handling and requirement for odd exposure sampling.
If a version has already been published, use a new version and update setup.py
and docs/conf.py together.

Test installation from the actual tag in a fresh environment:

```bash
python -m venv /tmp/bruce-tag-check
source /tmp/bruce-tag-check/bin/activate
python -m pip install --upgrade pip
python -m pip install 'bruce[analysis,gui] @ git+https://github.com/samgill844/bruce.git@v1.0.0'
```

Submit the public repository URL and paper/paper.md to JOSS. Check that the public
commit history, research-use evidence and licensing meet the journal's current
criteria. A DOI archive is normally prepared for the final reviewed version at
the editor's request; create the Zenodo archive from that version, and ensure
its author list and version agree with the paper. A release or DOI by itself
does not establish editorial eligibility.

If source distributions or wheels are wanted:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Test installation from the source archive. C wheels are platform-specific; a
wheel built on one machine is not automatically portable to other platforms
or OpenMP installations. Nothing in this guide uploads or publishes a package.
