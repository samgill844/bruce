# BRUCE JOSS manuscript

This directory contains a first author draft for the *Journal of Open Source Software*. The manuscript follows the current [JOSS paper format](https://joss.readthedocs.io/en/latest/paper.html): YAML metadata, a short research-software article and the required substantive sections. It is approximately 1,735 words including figure captions, within the current 750–1,750-word range. See [AUTHOR_NOTES.md](AUTHOR_NOTES.md) for items that require author review before submission.

## Main files

- [paper.md](paper.md): the editable JOSS manuscript, with Samuel Gill's supplied name, affiliation and ORCID.
- [paper.bib](paper.bib): 20 literature/software references with DOI or source links and full journal names.
- [output/paper-preview.pdf](output/paper-preview.pdf): locally typeset reading copy. This is **not the official JOSS proof**.
- [figures/](figures/): a single-panel synthetic transit-fit figure and a measured likelihood-throughput figure, supplied as PNG and vector PDF.
- [benchmarks/results.json](benchmarks/results.json): raw timing repetitions, settings, source hashes and machine details.
- [benchmarks/synthetic_transit.csv](benchmarks/synthetic_transit.csv) and [synthetic_fit.json](benchmarks/synthetic_fit.json): reproducible demonstration data and fitted parameters.

The synthetic transit is generated and fitted with BRUCE itself. The template-search and alias-resolution use cases are illustrated by references to published recovery studies. It demonstrates a workflow; it is not an independent test of transit physics. No actual astronomical target or survey result is claimed. The benchmark compares BRUCE operations/thread counts, not competing software.

## Reproduce the measurements and figures

Run from the **repository root**. Install BRUCE's analysis dependencies first. The manuscript tools additionally require the packages in `paper/requirements-paper.txt`.

```bash
python setup.py build_ext --force --build-lib /tmp/bruce-paper-lib --build-temp /tmp/bruce-paper-build
PYTHONPATH=/tmp/bruce-paper-lib:src OMP_NUM_THREADS=1 python paper/benchmarks/benchmark.py
PYTHONPATH=/tmp/bruce-paper-lib:src MPLCONFIGDIR=/tmp/bruce-paper-mpl OMP_NUM_THREADS=1 python paper/scripts/make_figures.py
python paper/scripts/check_paper.py
```

The recorded draft measurements were made on 3 November 2026 with Apple Clang 21.0.0 and OpenMP, using the options in [benchmarks/build.log](benchmarks/build.log). The benchmark driver launches separate worker processes with one and four OpenMP threads, disables dynamic OpenMP team sizing and limits BLAS threads. Each operation is warmed up, batch sizes are calibrated, and seven repeated batches are retained. The performance figure shows reciprocal median call times for the direct C likelihood, with bounds from the reciprocal timing quartiles; the right panel shows one-thread/four-thread timing ratios. Each evaluation includes the complete light curve and likelihood reduction. Timing covers public Python wrapper calls and their numerical work, but excludes imports, construction of input arrays, file/network I/O and optimiser execution. The combined likelihood path is checked against a NumPy reduction before timings are accepted.

All cases use `period=3` days, `radius_1=0.1`, `k=0.1`, `incl=arccos(0.03)`, `w=0.7`, `c=0.7`, `alpha=0.4`, no third light, fixed flux errors of 0.001, zero jitter, `offset=0` and `accurate_tp=0`. Times span −0.15 to +0.15 days. The cases are instantaneous circular motion, instantaneous eccentric motion (`e=0.4`), and circular motion integrated over 2 minutes with 11 subsamples. The seed is 42. Re-running overwrites the raw results, so update numerical claims in `paper.md` if the results change.

Hardware fields in the retained JSON were completed from a separately authorised read-only `sysctl` query because the sandbox prevented the worker from reading them. The benchmark records the working tree as dirty and stores source hashes: the Git commit alone does not identify the measured code. Commit the final measured source and repeat measurements for the release submission. The manuscript's timing claims use this particular OpenMP build; do not treat a serial build as equivalent.

The benchmark does not establish speed relative to `batman`, `ellc` or PyTransit. Such comparisons need matched physical and numerical settings and accuracy checks. It also does not measure posterior sampling throughput, archive processing or full template-search performance. A higher thread count can increase overhead for small workloads, and CPU load and thermal state affect timings.

## Check the source and numerical routines

```bash
python paper/scripts/check_paper.py
PYTHONPATH=/tmp/bruce-paper-lib:src MPLCONFIGDIR=/tmp/bruce-paper-mpl OMP_NUM_THREADS=1 python -m unittest discover -s tests/regression -v
```

The manuscript checker verifies metadata, required section headings, citation keys, image files, cross-references, approximate word count and benchmark source hashes. It is not a substitute for editorial or scientific review. The current regression suite has 15 tests, including independent bracketed Kepler comparisons and an RV integration check.

## Render a local reading copy

The local preview uses installed `wkhtmltopdf` plus Python Markdown and bibliography processing. Equations are rendered locally with Matplotlib; no external JavaScript service is needed.

```bash
MPLCONFIGDIR=/tmp/bruce-paper-mpl python paper/scripts/render_preview.py
pdftoppm -scale-to 1500 -png paper/output/paper-preview.pdf /tmp/bruce-paper-page
```

This renderer is intentionally separate from the submission source. Its abbreviated bibliography display and typography are for reading only. `paper.md` and `paper.bib` remain the authoritative input for the journal build. No hand-maintained LaTeX manuscript is needed. Pandoc, Docker/Inara and `latexmk` were unavailable in the drafting environment; the official typesetting pipeline has therefore **not been run locally**.

## Build the official JOSS proof

JOSS documents the following [Inara container workflow](https://joss.readthedocs.io/en/latest/paper.html#docker). With Docker installed, run from the repository root:

```bash
docker run --rm \
  --volume "$PWD/paper:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

This should produce `paper/paper.pdf` using the journal's own templates. Check that proof, its references, equations and figures before submitting. The journal also supports an [Open Journals GitHub Action](https://github.com/openjournals/openjournals-draft-action); no remote workflow has been installed or run by this draft setup.

## Replace or extend the figures

The single-panel transit-fit figure minimises the public direct C negative-log-likelihood with Nelder-Mead; `benchmarks/synthetic_c_likelihood_fit.csv` and `.json` retain its data, parameters and objective value. The original fit/residual example uses least squares.

The single-panel transit-fit figure is `figures/use_cases.png` (also PDF). The original fit/residual plot remains available as `figures/synthetic_transit.png`. The earlier nightly simulation, search scores and alias checks are retained in `benchmarks/synthetic_ngts.csv`, `synthetic_template_scores.csv` and `synthetic_search.json` as supplementary draft artefacts; they are no longer plotted or regenerated by `make_figures.py`.

Edit `scripts/make_figures.py` to change styling, or replace the corresponding files in `figures/` while retaining the labels `fig:transit` and `fig:performance` in the manuscript. A real TESS event and associated ground-based coverage would strengthen the research-use discussion. If replacing the synthetic demonstration, provide the source data identifiers, processing choices, reproducible analysis and relevant archive acknowledgements. Do not label a synthetic example as observed data.
