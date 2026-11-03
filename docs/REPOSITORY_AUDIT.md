# Repository audit

Audit date: 3 November 2026. The checkout contained substantial pre-existing edits and untracked research code. Those were preserved; this report describes targeted maintenance rather than validation of all scientific workflows.

## Size and generated products

Initial disk usage was approximately 311 MB, of which 246 MB was `.git`, 51 MB examples, 8.8 MB images and 5.3 MB web assets. No history was rewritten. Removing files in a new commit does not remove historical blobs.

No FITS files or `mastDownload` contents were tracked. The following local products were retained; they may contain useful observations or derived results. Both discovered `mastDownload` directories were empty at audit time.

| Local FITS product | Bytes |
| --- | ---: |
| `examples/duotransits/13093071/TIC-13093071_ALIASES.fits` | 17,280 |
| `examples/duotransits/186072225/TIC-186072225_ALIASES.fits` | 8,640 |
| `examples/solved_systems/hlsp_tess-spoc_tess_phot_0000000412635642-s0064_tess_v1_lc.fits` | 1,192,320 |
| `examples/solved_systems/hlsp_tess-spoc_tess_phot_0000000412635642-s0065_tess_v1_lc.fits` | 1,232,640 |
| `examples/solved_systems/tess2023096110322-s0064-0000000412635642-0257-s_lc.fits` | 1,969,920 |
| `examples/solved_systems/tess2023124020739-s0065-0000000412635642-0259-s_lc.fits` | 2,039,040 |
| `examples/solved_systems/tess2026005125623-s0099-0000000412635642-0300-s_lc.fits` | 2,021,760 |
| `examples/solved_systems/tess2026033082000-s0100-0000000412635642-0302-s_lc.fits` | 1,935,360 |
| `examples/solved_systems/tess2026060005000-s0101-0000000412635642-0303-s_lc.fits` | 1,912,320 |

New ignore rules cover FITS products, MAST directories, HDF chains, Python caches, shared objects and compiled experiments. Tests are no longer globally ignored. Existing interactive scripts under `tests/` are retained; the automated suite is scoped to `tests/regression/`.

Removed unchanged tracked artefacts:

- `src/bruce_c/.DS_Store`
- `src/bruce_c/a.out`
- `src/bruce_c/simd/avx`
- `src/bruce_c/simd/avx_loglike`
- `src/bruce_c/simd/simd_example`
- `src/bruce_c/simd/timing_aligned`
- `src/utils/.DS_Store`

Modified `.DS_Store` files were retained to avoid discarding pre-existing changes. The source files for compiled SIMD experiments remain available. Paper figures, example notebooks, CSV observations, archived scripts, `web/`, the untracked scratch script and GIF were retained: absence of imports is insufficient evidence that research assets are unused.

## Fixes and optimisation

- C likelihood evaluation now reads converted, contiguous double-precision arrays, rather than interpreting original integer/float buffers as doubles. Conversion failure paths release owned references.
- Several legacy C wrappers now require one-dimensional arrays, preventing allocation sizes from disagreeing with loop bounds.
- The light-curve likelihood wrapper checks the error-array conversion correctly and releases earlier conversions on failure.
- Fixed binning rejects empty, unequal, unsorted or non-finite time arrays and invalid bin widths. Adaptive binning rejects empty or unequal inputs and releases conversions when no bins are produced.
- Phase dispersion validates conversions, allocation results, data lengths, peak indices and positive finite periods before entering the numerical routine.
- Time-gap segmentation uses adjacent differences and supports empty and singleton arrays.
- Detrending calls the intended SciPy median filter rather than the shadowing time-domain wrapper, honours `splinesmooth`, and reports unknown methods.
- The NGTS filter-window helper now requires an explicit theoretical noise level instead of referencing undefined global catalogue/noise arrays.
- The Python likelihood uses NumPy operations instead of a per-observation Python loop; its existing finite-contribution policy is retained.
- The single-template S/N guard checks the actual `radius_1` argument.
- Period-resolver instances no longer share a mutable default event list.
- Package submodules, TTV dependencies and archive loading are deferred until used.
- The NASA archive reader uses the same cache directory as the downloader; the downloader no longer constructs shell commands.
- Build flags preserve IEEE NaN/Inf behaviour, avoid a forced x86 architecture, support both common Homebrew prefixes and allow serial builds. Core dependencies and optional analysis/GUI dependencies are declared.
- The broken `lcmatch` entry point was removed; its archived implementation references obsolete APIs and is not silently restored as a supported command.

## Remaining limitations

- The untracked TESS scratch script had three malformed row-index expressions; these were corrected, and its example download call now runs only when executed directly.
- No scientific formulae, empirical coefficients, likelihood signs or conclusions were intentionally changed. The low-level C likelihood wrappers return negative log likelihood; the Python sampler returns log likelihood. Downstream code must account for this existing difference.
- `compute_dispersion` has its likelihood calculation commented out; the second returned array is zero-filled. It must not be interpreted as a computed chi-squared statistic.
- Batch template matching still uses a fixed width of 0.5 days in its Python setup. This needs a dedicated scientific review rather than an arbitrary replacement.
- Iterative Savitzky–Golay clipping can leave too few retained samples for the requested window; further detrending work needs explicit behaviour for short and heavily clipped data.
- Archive helpers retain installation-specific resources and shell-based retrieval in other functions. `data_retrieval.py` also retains a global `np.int` compatibility assignment. Live archive retrieval, network error handling, GUI tools, HDF resume workflows and the full transit-finding pipeline were not exercised.
- `tesstpf` contains a legacy `bruce2` import in one loading path. Archived utilities contain obsolete API references. These need workflow-specific examples before safe removal or migration.
- C kernels still merit further review for extreme bin counts, allocation failures, invalid orbital parameters and sorted-data assumptions. The added checks are targeted, not a complete memory-safety audit.
- The empirical radius-to-mass mapping is branch-dependent and is not a globally unique inverse. Its scientific assumptions were retained.
- No catalogue downloads, history cleanup or deletion of observation data were performed.

## Verification

The extension compiled locally both with Clang/Homebrew libomp and with `BRUCE_OPENMP=0`. All nine regression tests passed in both builds. The deterministic suite exercises converted/strided likelihood inputs, shape rejection, bin validation, gap boundaries, filter/spline behaviour, flux conversions, lazy imports, vectorised likelihood equivalence and period-search bounds. See README for the build and test commands. Local third-party packages emitted NumPy ABI and pandas optional-dependency warnings; those environment issues were not repaired as part of this repository change.

The source tree and setup script parsed successfully with Python AST checks. `git diff --check` reports trailing whitespace in pre-existing working-tree edits; those unrelated edits were retained.
