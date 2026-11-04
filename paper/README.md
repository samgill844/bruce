# BRUCE JOSS manuscript

This directory contains a first author draft for the *Journal of Open Source Software*. The manuscript follows the current [JOSS paper format](https://joss.readthedocs.io/en/latest/paper.html): YAML metadata, a short research-software article and the required substantive sections. It is approximately 1,675 words including figure captions. The current JOSS target is 750–1,750 words; run the checker after editing.

## Main files

- [paper.md](paper.md): the editable JOSS manuscript, with Samuel Gill's supplied name, affiliation and ORCID.
- [paper.bib](paper.bib): 20 literature/software references with DOI or source links and full journal names.
- [output/paper-preview.pdf](output/paper-preview.pdf): locally typeset reading copy. This is **not the official JOSS proof**.
- [figures/](figures/): a two-panel matched-law BRUCE/ellc comparison and a combined likelihood-throughput/SPOCFIT figure, supplied as PNG and PDF.
- [benchmarks/results.json](benchmarks/results.json): raw timing repetitions, settings, source hashes and machine details.
- [benchmarks/synthetic_transit.csv](benchmarks/synthetic_transit.csv) and [synthetic_fit.json](benchmarks/synthetic_fit.json): reproducible demonstration data and fitted parameters.

The main figure compares BRUCE with numerical integration of the identical power-2 law in ellc. Supplementary synthetic transits are generated and fitted with BRUCE itself. The template-search and alias-resolution use cases are illustrated by references to published recovery studies. It demonstrates a workflow; it is not an independent test of transit physics. Figure 2 includes a screenshot of real TESS photometry for TIC 100504488; no new target characterisation or survey result is claimed. The comparison is instantaneous and covers one specified geometry; it is not a global error bound. The benchmark compares BRUCE operations/thread counts, not competing software.

## Reproduce the measurements and figures

Run from the **repository root**. Install BRUCE's analysis dependencies first. The manuscript tools additionally require the packages in `paper/requirements-paper.txt`.

```bash
python setup.py build_ext --force --build-lib /tmp/bruce-paper-lib --build-temp /tmp/bruce-paper-build
PYTHONPATH=/tmp/bruce-paper-lib:src OMP_NUM_THREADS=1 python paper/benchmarks/benchmark.py
PYTHONPATH=/tmp/bruce-paper-lib:src MPLCONFIGDIR=/tmp/bruce-paper-mpl OMP_NUM_THREADS=1 python paper/scripts/make_figures.py
python paper/scripts/check_paper.py
```

The recorded draft measurements were made on 3 November 2026 with Apple Clang 21.0.0 and OpenMP, using the options in [benchmarks/build.log](benchmarks/build.log). The benchmark driver launches separate worker processes with one and four OpenMP threads, disables dynamic OpenMP team sizing and limits BLAS threads. Each operation is warmed up, batch sizes are calibrated, and seven repeated batches are retained. The left panel of Figure 2 shows reciprocal median call times for the direct C likelihood, with bounds from the reciprocal timing quartiles. The right panel uses the author-supplied screenshot at `docs/_static/cli/spocfit.png`, showing TIC 100504488 in TESS sector 27. Four-thread measurements remain in the raw data and manuscript text. Each evaluation includes the complete light curve and likelihood reduction. Timing covers public Python wrapper calls and their numerical work, but excludes imports, construction of input arrays, file/network I/O and optimiser execution. The combined likelihood path is checked against a NumPy reduction before timings are accepted.

All cases use `period=3` days, `radius_1=0.1`, `k=0.1`, `incl=arccos(0.03)`, `w=0.7`, `c=0.7`, `alpha=0.4`, no third light, fixed flux errors of 0.001, zero jitter, `offset=0` and `accurate_tp=0`. Times span −0.15 to +0.15 days. The cases are instantaneous circular motion, instantaneous eccentric motion (`e=0.4`), and circular motion integrated over 2 minutes with 11 subsamples. The seed is 42. Re-running overwrites the raw results, so update numerical claims in `paper.md` if the results change.

Hardware fields in the retained JSON come from the earlier authorised read-only `sysctl` query on this machine; the benchmark was refreshed after correcting the likelihood-wrapper argument type. Pre-fix records are retained under `benchmarks/archive/pre_wrapper_fix/`. The benchmark records the working tree as dirty and stores source hashes: the Git commit alone does not identify the measured code. Commit the final measured source and repeat measurements for the release submission. The manuscript's timing claims use this particular OpenMP build; do not treat a serial build as equivalent.

The benchmark does not establish speed relative to `batman`, `ellc` or PyTransit. Such comparisons need matched physical and numerical settings and accuracy checks. It also does not measure posterior sampling throughput, archive processing or full template-search performance. A higher thread count can increase overhead for small workloads, and CPU load and thermal state affect timings.

## Check the source and numerical routines

```bash
python paper/scripts/check_paper.py
PYTHONPATH=/tmp/bruce-paper-lib:src MPLCONFIGDIR=/tmp/bruce-paper-mpl OMP_NUM_THREADS=1 python -m unittest discover -s tests/regression -v
```

The manuscript checker verifies metadata, required section headings, citation keys, image files, cross-references, approximate word count and benchmark source hashes. It is not a substitute for editorial or scientific review. The current regression suite has 21 tests when ellc is installed, including independent bracketed Kepler comparisons and an RV integration check.

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

This should produce `paper/paper.pdf` using the journal's own templates. Check that proof, its references, equations and figures before submitting. The journal also supports an [Open Journals GitHub Action](https://github.com/openjournals/openjournals-draft-action); the workflow in `.github/workflows/paper.yml` builds `paper/paper.md` and uploads the official-toolchain PDF as an artifact. It has not yet run remotely; check its result after pushing.

## Numerical comparison (main Figure 1)

`python paper/scripts/compare_ellc.py` generates `figures/ellc_comparison.png` and
`.pdf`, plus `benchmarks/ellc_comparison.csv` and `.json`. It uses the geometry of
Maxted & Gill (2019), Figure 3, and compares the **same power-2 law** in BRUCE and
ellc; it does not use a tabulated STAGGER atmosphere profile as the reference.
Parameters are Rstar/a=0.05, Rp/Rstar=0.1, b=0, e=0, P=1 day and instantaneous
sampling. Both implementations use c=0.715452 and alpha=0.643777, the CHEOPS
6000 K, log g=4.5, solar-metallicity coefficients tabulated by Maxted (2018).
The script contains these values explicitly and does not import pycheops or
access the network. No parameters are fitted to reduce the residuals.

The reference uses ellc 1.8.15 with spherical bodies and very-fine integration.
The maximum BRUCE/reference difference is 74.0 ppm and the RMS is 20.4 ppm.
Ellc's fine/very-fine grid difference is at most 0.58 ppm in this case. The
sparse/reference residual is plotted for context. The original K2 expression
is unchanged as requested. Raw model arrays, numerical settings, versions and
source hashes are retained. The optional independent regression test checks this
case against ellc; it does not establish accuracy across all radii or inclinations.

Finite exposures now require an odd `noversample >= 3`, with default 11, in both
light-curve/likelihood and template-matching entry points. Explicit even values
raise ValueError. Separate symmetry tests exercise finite exposures. Records from
before this change remain under `benchmarks/archive/pre_odd_sampling/`.

## Supplementary figures

The single-panel transit-fit figure minimises the public direct C negative-log-likelihood with Nelder-Mead; `benchmarks/synthetic_c_likelihood_fit.csv` and `.json` retain its data, parameters and objective value. The original fit/residual example uses least squares.

The earlier single-panel transit-fit figure is retained as `figures/use_cases.png` (also PDF), but is no longer the manuscript figure. The original fit/residual plot remains available as `figures/synthetic_transit.png`. The earlier nightly simulation, search scores and alias checks are retained in `benchmarks/synthetic_ngts.csv`, `synthetic_template_scores.csv` and `synthetic_search.json` as supplementary draft artefacts; they are no longer plotted or regenerated by `make_figures.py`.

Edit `scripts/make_figures.py` to change styling, or replace the corresponding files in `figures/` while retaining the labels `fig:transit` and `fig:performance` in the manuscript. A real TESS event and associated ground-based coverage would strengthen the research-use discussion. If replacing the synthetic demonstration, provide the source data identifiers, processing choices, reproducible analysis and relevant archive acknowledgements. Do not label a synthetic example as observed data.

## Licence and likelihood-wrapper correction

The package now uses GPL-3.0-or-later, with ellc attribution retained in the root
`NOTICE.md`. The direct light-curve likelihood wrapper parses `offset` into an
integer, matching the Python C API. Regression tests compare both `offset=0`
and `offset=1` against NumPy, with jitter, eccentricity, exposure integration,
float32 inputs and strided arrays. The normalised mode includes the
variance-dependent logarithmic term; the additive Gaussian constant is omitted.
The current benchmark continues to use `offset=0` for continuity with the earlier
measurements. No post-fix speed improvement is inferred from the difference
between these two local timing runs.

For tagged releases and optional PyPI publication, see [RELEASING.md](../RELEASING.md). No tag, remote release or upload was created by this work.

## Final preparation record

See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for checks completed on 4 November 2026, remaining author confirmations, and the official-build workflow.
