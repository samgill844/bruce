# JOSS submission preparation — 4 November 2026

The local reading PDF has been regenerated and visually inspected. This is a
preparation record, not an editorial eligibility decision or an official JOSS
compilation result.

## Completed locally

- Manuscript: required section headings, YAML metadata, 20 citation keys, figure
  assets and cross-references pass the source checker; approximately 1,675 words.
- Corrected grammar and the NumPy C API hyperlink; clarified why the combined
  C/Python workflow is a separate package and cited the coefficient source.
- Four-page local preview inspected, including both figures and references.
- Fresh C extension build: 21 regression tests pass, including the optional
  ellc matched-law comparison. The benchmark's recorded source hashes match;
  benchmark timings were not rerun or reinterpreted as a new speed improvement.
- A wheel built from the generated source archive installs into a separate
  directory and passes the same 21 tests with existing environment dependencies.
  This does not establish a fresh dependency resolution on every platform.
- Sphinx builds with warnings treated as errors. Internal links, images and
  anchors checked across 21 HTML pages. Root README links repaired.
- The source archive contains C headers, licence notices, docs and screenshots,
  and excludes generated HTML and FITS downloads.
- CONTRIBUTING.md provides contribution, support and validation instructions.
  RELEASING.md describes GitHub releases and installation from the release tag.
- Added test/documentation CI and a separate Open Journals compilation workflow.
  Their remote results remain unverified until the changes are pushed.
- Replaced obsolete QtGui widget aliases with QtWidgets in spocfit and tesstpf
  and repaired their local-FITS hand-off.
  Graphical windows and live MAST retrieval were not rerun in this audit.

## Before submission

- Commit and push all source/documentation changes, including currently untracked
  screenshots, CLI guides and workflow files. Run both workflows on the commit
  to be released and inspect the JOSS PDF artifact. Docker/Pandoc are unavailable
  locally, so the official journal build remains unverified here.
- Create your versioned GitHub release after checking that commit. Ask a colleague
  to install from its tag in a fresh environment and complete a worked example.
- Manually confirm the documented GUI workflows on the release environment.
  The local FITS hand-off from tesstpf to spocfit is now repaired and has two
  offline regression tests preserving target identity, BTJD times, flux and
  uncertainties; the graphical launch still needs a manual check.
- Confirm that the repository has been public for more than six months and
  preserves its iterative development history. Local commits span October 2025
  onward, but commit dates alone do not prove when code became public.
- Confirm and make explicit which research workflows actually used BRUCE. The
  cited recovery papers currently establish scientific context; the manuscript
  does not claim every cited paper used the current package. JOSS expects
  demonstrated research use, at least by the developers, not only future aims.
- Confirm the author list, affiliation, ORCID, grant and other funding/COI
  disclosures, including any sponsor involvement statement required by JOSS.
- Complete the AI disclosure from your own records: tools/models and versions
  where known, scope, and your review/validation of all assisted outputs and
  responsibility for the core design decisions. Do not assert checks you have
  not personally performed. Write your own exchanges with editors/reviewers in
  accordance with JOSS's current AI policy.
- Submit paper/paper.md and the public repository/release details. Prepare a
  permanent DOI archive of the final reviewed version when requested by the
  editor. PyPI publication is optional.

## References for the author

- [JOSS submission requirements and screening](https://joss.readthedocs.io/en/latest/submitting.html)
- [JOSS paper format and compilation](https://joss.readthedocs.io/en/latest/paper.html)
- [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)
