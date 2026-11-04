"""Generate source-derived Sphinx signatures without importing optional modules."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/api'
# Common parameter meanings; per-function overrides below take precedence.
ARGS = {
 't': 'One-dimensional NumPy time array, in days; a common time system with t_zero.',
 'time': 'Sorted one-dimensional NumPy time array, in days.',
 'times': 'One-dimensional time array in the same units as epoch and period.',
 'flux': 'Flux array on the time grid; relative flux for models.',
 'flux_err': 'Positive one-sigma flux uncertainties, in flux units.',
 't_zero': 'Primary mid-transit epoch in the same time system as t.',
 'period': 'Orbital period in days.', 'P': 'Orbital period in days.',
 'radius_1': 'Scaled primary radius R_star/a.', 'k': 'Companion-to-primary radius ratio R_companion/R_star.',
 'incl': 'Orbital inclination in radians.', 'e': 'Eccentricity, with 0 <= e < 1.',
 'w': 'Argument of periastron in radians.',
 'c': 'Power-2 limb-darkening coefficient (or linear coefficient for ld_law=1).',
 'alpha': 'Power-2 limb-darkening exponent.',
 'cadence': 'Full exposure duration in days; zero means instantaneous evaluation.',
 'noversample': 'Exposure subsample count; odd and at least 3 when cadence is positive. Default 11.',
 'light_3': 'Contaminating light divided by target light.',
 'ld_law': 'Law selector: 2 power-2, 1 linear, 0 uniform, -1 box with depth k, -2 single-event power-2.',
 'accurate_tp': '1 minimises projected separation for epoch-to-periastron conversion; 0 uses the edge-on conjunction approximation.',
 'jitter': 'Additional independent scatter added in quadrature to uncertainties; same units as the observations.',
 'offset': 'Include the variance-dependent logarithmic term if 1/True; omit if 0/False. Gaussian additive constant is omitted.',
 'K1': 'Primary RV semi-amplitude in the chosen velocity unit.',
 'K2': 'Secondary RV semi-amplitude in the same velocity unit as K1.',
 'V0': 'Systemic velocity in the same units as K1.',
 'rv': 'Observed primary velocities; None requests model evaluation.',
 'rv_err': 'Positive one-sigma uncertainties on primary velocities.',
 'rv1': 'Observed primary velocities; None requests both model curves.',
 'rv2': 'Observed secondary velocities on the same time grid.',
 'rv1_err': 'Positive primary-velocity uncertainties.', 'rv2_err': 'Positive secondary-velocity uncertainties.',
 'b': 'Circular-orbit impact parameter in primary-radius units.',
 'normalisation_model': 'Baseline flux array on the observation grid; transit flux multiplies this baseline.',
 'time_step': 'Trial-epoch spacing in days; None selects circular transit width/20.',
 'time_trial': 'Explicit NumPy array of trial epochs; overrides automatic epoch-grid generation.',
 'bin_size': 'Positive time-window or target bin width in the same units as time.',
 'gap': 'Time gap used to split the adaptive binning into separate segments.',
 'target_bin_size': 'Target time width of each adaptive bin.',
 'min_acceptable_bin_size': 'Minimum accepted time span of an adaptive bin.',
 'width': 'Full proximity window; trial epochs within half this width of data are selected.',
 'x': 'Input array; see the function description.', 'dx_lim': 'Positive gap threshold in x units; gaps >= this value split groups.',
 'mags': 'Differential magnitudes with flux=10**(-mags/2.5).',
 'mags_err': 'One-sigma magnitude uncertainties.', 'epoch': 'Reference epoch in the same units as times.',
 'phase_offset': 'Phase shift defining the wrap interval [-phase_offset, 1-phase_offset).',
 'unique_entries': 'If True, retain the last row per planet name from the cached archive table.',
 'user': 'Optional username passed to the local institutional implementation; None uses its local default.',
 'id': 'Object identifier passed to SIMBAD.', 'tic_id': 'TESS Input Catalog identifier; None returns None where supported.',
 'data_release': 'Gaia release number used to select the Gaia source table.',
 'val': 'Scalar parameter value.', 'low': 'Inclusive lower bound.', 'high': 'Inclusive upper bound.',
 'reference': 'Normal-prior mean.', 'reference_err': 'Positive normal-prior standard deviation.',
 'y': 'Observed values.', 'yerr': 'Positive one-sigma measurement uncertainties.',
 'model': 'Predicted values matching the observations.',
 'fs': 'sqrt(e)*sin(w).', 'fc': 'sqrt(e)*cos(w).',
 'h1': 'Power-2 transformed coefficient 1-c*(1-0.5**alpha).',
 'h2': 'Power-2 transformed coefficient c*0.5**alpha.',
 'sampler': 'An emcee EnsembleSampler with a stored chain.',
 'burn_in': 'Number of initial steps discarded from each walker.',
 'theta_names': 'Parameter labels in chain-column order.',
 'blob_names': 'Optional labels for auxiliary values (blobs) returned by the target function.',
 'verbose': 'Print progress or results when True.', 'name': 'Label included in the printed report.',
 'llf': 'Maximised log likelihood (not negative log likelihood).', 'nobs': 'Number of observations.',
 'df_modelwc': 'Number of fitted parameters, including any constant.',
 'theta': 'One-dimensional initial parameter vector.', 'nwalkers': 'Number of ensemble walkers.',
 'ndim': 'Number of parameters.', 'func': 'Callable returning log posterior (larger is better), optionally with emcee blobs.',
 'nsteps': 'Number of MCMC steps per walker.', 'moves': 'Weighted emcee move objects; source defaults use DE and DESnooker moves.',
 'nthreads': 'Number of multiprocess worker processes, not OpenMP threads.',
 'progress': 'Show the emcee progress indicator.',
 'refit': 'True resets the backend and runs a new chain; False constructs a sampler around the existing backend without running.',
 'backend': 'HDF5 backend filename, or None for an in-memory chain. refit=True resets an existing file.',
 'args': 'Additional positional arguments passed by emcee to func.',
 'vals': 'Three values [central value, lower uncertainty, upper uncertainty].',
 'dp': 'Initial number of decimal places.', 'best_step': 'Best-fit parameter vector.',
 'low_error': 'Lower uncertainty vector.', 'high_error': 'Upper uncertainty vector.',
 'planets_params': 'List of dictionaries with t_zero, period, radius_1, k, incl, e, w and optional label; same conventions as lc.',
 'line_alpha': 'Opacity of the trajectory lines.', 'zoom_primary': 'Limit the sky-plane view to +/-1.5 stellar radii if True.',
 'save_gif': 'Save an animated GIF using Pillow if True.', 'gif_path': 'Destination GIF filename.', 'fps': 'Saved GIF frames per second.',
 'q': 'Companion-to-primary mass ratio.',
 'windows': 'Candidate median-filter time windows; default scans 10.49 down to 0.5 days.',
 'tolerence': 'Allowed fractional excess over theoretical_noise (historical spelling).',
 'theoretical_noise': 'Required positive target scatter in ppm per 30-minute integration.',
 'method': 'Trend estimator: poly1d, spline or savgol.',
 'max_median': '0 disables preprocessing; 1 applies maximum then median filter; 2 applies median then maximum filter.',
 'npoly': 'Polynomial degree for poly1d.', 'Nmaxfilter': 'Maximum-filter window size in samples.',
 'Nmedianfilter': 'Median-filter window size in samples.', 'splinesmooth': 'UnivariateSpline smoothing factor.',
 'SG_window_length': 'Savitzky-Golay window length in samples; choose an odd length larger than SG_polyorder.',
 'SG_polyorder': 'Savitzky-Golay polynomial order.', 'SG_deriv': 'Derivative order; zero returns the smoothed trend.',
 'SG_delta': 'Sample spacing for Savitzky-Golay derivatives.', 'SG_iter': 'Number of clipping/filter passes.',
 'SG_sigma': 'Residual clipping threshold in standard-deviation units.',
 'filename': 'Path to an NGTS FITS product with the expected ACTIONID, CLEAN and photometry columns.',
 'optimal_index': 'FITS table extension containing the selected photometry.', 'action_id': 'NGTS action identifier to plot.',
}
# Summary and exact return contract for every documented function.
INFO = {
 'lc': ('Evaluate an analytical primary transit, or its C negative log likelihood.', 'Flux array, or scalar negative log likelihood when flux and flux_err are supplied.'),
 'rv1': ('Evaluate a single-lined Keplerian RV model or its likelihood.', 'Primary velocity array, or scalar negative log likelihood with observations.'),
 'rv2': ('Evaluate a double-lined Keplerian RV model or its likelihood.', 'Tuple (primary_velocity, secondary_velocity), or scalar negative log likelihood with observations.'),
 'transit_width': ('Calculate the total circular transit duration.', 'Duration in the same time units as period; assumes a transiting circular geometry.'),
 'stellar_density': ('Calculate mean stellar density from scaled radius with the implemented eccentric correction.', 'Mean density in solar units.'),
 'template_match_lightcurve': ('Compare fixed transit templates with a supplied baseline at trial epochs.', 'Tuple (time_trial, score). Scalar geometry gives a 1D score; array geometry gives axes (epoch, radius_1, k, incl). Score is 2*(logL_transit-logL_baseline).'),
 'animate_orbit': ('Display independent Keplerian orbits and optionally save a GIF.', 'None. Opens a Matplotlib figure and calls show(); optionally writes gif_path.'),
 'median_filter': ('Evaluate a time-window median trend at each input timestamp.', 'Filtered flux array on the input grid.'),
 'convolve_1d': ('Evaluate a time-window mean trend at each input timestamp.', 'Filtered flux array on the input grid.'),
 'bin_data': ('Bin sorted photometry into approximately bin_size-wide bins; empty bins are omitted.', 'Tuple (bin_mid_times, mean_flux, flux_standard_error, counts); uncertainties are estimated from scatter, not supplied errors.'),
 'bin_data_adaptive': ('Bin sorted data in segments separated by gaps.', 'Dictionary with NumPy arrays t_start, t_end, flux, flux_err and count; edges are observed segment/bin endpoints.'),
 'check_proximity_of_timestamps': ('Identify trial epochs close to at least one observed timestamp.', 'Boolean mask with the shape of time_trial.'),
 'find_nights_from_data': ('Split sorted timestamps at gaps >= dx_lim.', 'List of non-empty integer index arrays into x.'),
 'mags_to_flux': ('Convert differential magnitudes and errors to relative flux.', 'Tuple (flux, flux_err).'),
 'flux_to_mags': ('Convert strictly positive relative flux and errors to magnitudes.', 'Tuple (magnitudes, magnitude_errors).'),
 'phase_times': ('Wrap timestamps into orbital phase.', 'Phase array in [-phase_offset, 1-phase_offset).'),
 'flatten_data_with_function': ('Estimate a trend with optional maximum/median preprocessing. This returns the trend, not divided flux.', 'Trend array on the input grid; divide the original flux by it to flatten.'),
 'find_optimal_ngts_filter_window': ('Select a window using measured versus target 30-minute scatter.', 'Tuple (window, time, flux, flux_err, counts, trend); uses a two-day fallback if no trial meets the target.'),
 'download_NASA_exoplanet_archive': ('Download the NASA ps table into ~/.cache/bruce/confirmed_planets.csv; network access required.', 'None; writes the cache file.'),
 'load_NASA_exoplanet_archive': ('Read the NASA ps cache, downloading it if absent.', 'Astropy Table; unique_entries keeps one row per pl_name.'),
 'query_simbad': ('Query SIMBAD object identifiers; network access required.', 'Astroquery SIMBAD result table, or None if no match.'),
 'download_tois_ctois': ('Optional local-only institutional TOI/CTOI downloader; implementation is not distributed.', 'Delegated to the local implementation; raises RuntimeError when it is not configured.'),
 'load_toi_table': ('Read the previously downloaded ~/.cache/bruce2/tois.dat file.', 'Astropy Table; no automatic download.'),
 'load_ctoi_table': ('Read the previously downloaded ~/.cache/bruce2/ctois.dat file.', 'Astropy Table; no automatic download.'),
 'load_tic8_info_for_target': ('Optional local-only institutional TIC query; implementation is not distributed.', 'Local implementation result, or None when tic_id is None; raises RuntimeError when unconfigured.'),
 'load_gaia_table': ('Query Gaia within 30 arcsec of an institutionally retrieved TIC position.', 'Tuple (TIC table, first Gaia result row); this does not identify the closest or correct counterpart automatically.'),
 'prior_uniform': ('Evaluate a bounded uniform log prior.', '0 within inclusive bounds; -inf outside.'),
 'prior_normal': ('Evaluate a normal log prior without the additive Gaussian constant.', 'Scalar log prior: -0.5*((val-reference)/reference_err)**2 - log(reference_err).'),
 'prior_bounded_normal': ('Evaluate the normal log prior within uniform bounds.', 'Scalar log prior, or -inf outside the bounds.'),
 'loglike_': ('Evaluate pointwise Gaussian log likelihoods.', 'Array of log likelihood contributions, including non-finite entries if present.'),
 'loglike': ('Sum finite pointwise Gaussian log likelihoods. Validate the observations and model first.', 'Scalar log likelihood; non-finite contributions are discarded.'),
 'fsfc_to_ew': ('Convert sqrt(e)*sin(w), sqrt(e)*cos(w) to orbital elements.', 'Tuple (e, w), with w in radians.'),
 'ew_to_fsfc': ('Convert eccentricity and periastron argument to sampling coordinates.', 'Tuple (sqrt(e)*sin(w), sqrt(e)*cos(w)).'),
 'ca_to_h1h2': ('Transform power-2 coefficients into h1/h2 coordinates.', 'Tuple (h1, h2).'),
 'h1h2_to_ca': ('Invert the power-2 h1/h2 transformation.', 'Tuple (c, alpha). Inputs must produce a valid logarithm.'),
 'incl_from_radius_1_b': ('Convert circular impact parameter to inclination.', 'arccos(radius_1*b), in radians; requires abs(radius_1*b)<=1.'),
 'get_sampler_report': ('Summarise a stored emcee chain around its maximum-log-probability sample.', 'Dictionary with out_text, pars, samples, samples_raw, logs, theta_names, best_step and burn_in. Error bounds use 16th/84th percentiles relative to best_step.'),
 'bic': ('Calculate the Bayesian information criterion.', '-2*llf + log(nobs)*df_modelwc.'),
 'create_starting_positions': ('Perturb a parameter vector with Gaussian scatter of absolute scale 1e-4.', 'Array of shape (nwalkers, ndim). Check prior support and ensemble independence yourself.'),
 'intilise_and_run_sampler': ('Construct and optionally run emcee with a multiprocess pool. The misspelling is part of the current API.', 'emcee EnsembleSampler. Uses 2*len(theta) walkers; a resumed backend is not advanced when refit=False. Use a __main__ guard in scripts.'),
 'analyse_sampler': ('Print a chain summary; uncertainty widths are computed about the median while the displayed central value is the best sample.', 'Ten-item tuple (samples, logs, best_idx, best_step, low_err, high_err, blobs, best_blob, blob_low_err, blob_high_err); missing blob outputs are -1.'),
 'corner_sampler': ('Plot a corner diagram for retained chain parameters.', 'Matplotlib Figure.'),
 'corner_sampler_blobs': ('Plot a corner diagram for auxiliary blob values.', 'Matplotlib Figure.'),
 'get_assymetric_latex_format': ('Format [value, lower_error, upper_error] as a LaTeX string; historical spelling retained.', 'LaTeX string.'),
 'print_fmt_latex': ('Print a named LaTeX value/error table row.', 'None; prints to standard output.'),
 'print_vals_from_sampler_latex': ('Print a LaTeX row for every parameter.', 'None; prints to standard output.'),
 'load_data': ('Load and normalise NGTS action photometry into self.data.', 'None; updates the instance table.'),
 'plot_action_summary_matplotlib': ('Plot one action using Matplotlib.', 'See the implementation; primarily creates a diagnostic plot.'),
 'plot_action_summary_pyqtgraph': ('Display one action with the optional Qt/pyqtgraph interface.', 'See the implementation; requires a working Qt GUI environment.'),
}

GROUPS = {
 'binarystar': ('Binary-star and transit models', [('bruce.binarystar', 'binarystar/binarystar.py', None)]),
 'template_match': ('Transit template matching', [('bruce.template_match', 'template_match/template_match.py', ['template_match_lightcurve'])]),
 'visual': ('Orbit visualisation', [('bruce.visual', 'visual/animate_orbits.py', None)]),
 'data': ('Data utilities', [('bruce.data', 'data/data_processing.py', ['median_filter','convolve_1d','bin_data','bin_data_adaptive','check_proximity_of_timestamps','find_nights_from_data','mags_to_flux','flux_to_mags','phase_times','flatten_data_with_function']), ('bruce.data.data_processing','data/data_processing.py',['find_optimal_ngts_filter_window']), ('bruce.data.nasa_archive', 'data/nasa_archive.py', None)]),
 'sampler': ('Sampling and likelihood utilities', [('bruce.sampler', 'sampler/sampler.py', None)]),
}

def signature(node):
    # Preserve defaults verbatim, but abbreviate large array/move expressions.
    args = node.args.args
    defaults = [None]*(len(args)-len(node.args.defaults)) + list(node.args.defaults)
    parts = []
    for arg, default in zip(args, defaults):
        if arg.arg == 'self':
            continue
        text = arg.arg
        if default is not None:
            value = ast.unparse(default)
            if len(value)>55 or isinstance(default, (ast.Call, ast.List, ast.Subscript)):
                value = '...'
            text += '=' + value
        parts.append(text)
    return '(' + ', '.join(parts) + ')'

def entry(module, node, kind='function'):
    name = node.name
    description, returns = INFO[name]
    lines = [f'.. py:{kind}:: {module}.{name}{signature(node)}', '', '   '+description, '']
    for arg in node.args.args:
        if arg.arg == 'self':
            continue
        desc = ARGS[arg.arg]
        if arg.arg == 't' and name == 'lc':
            desc += ' Default np.linspace(-0.2, 0.2, 100).'
        elif arg.arg == 't' and name in ('rv1', 'rv2'):
            desc += ' Default np.linspace(0, 1, 100).'
        if name == 'template_match_lightcurve' and arg.arg in ('radius_1', 'k', 'incl'):
            desc += ' For batch searches, supply all three geometry parameters as one-dimensional NumPy arrays.'
        if name in ('get_assymetric_latex_format', 'print_fmt_latex') and arg.arg == 'x':
            desc = 'Three values [central value, lower uncertainty, upper uncertainty].'
        if name == 'animate_orbit' and arg.arg == 't':
            desc = 'Time samples in days; default np.linspace(-0.1, 0.1, 1000). One frame per sample.'
        if name == 'find_nights_from_data' and arg.arg == 'x':
            desc = 'Sorted one-dimensional timestamps.'
        lines += [f'   :param {arg.arg}: {desc}']
    lines += ['   :returns: '+returns, '']
    return lines

def main():
    for key, (title, modules) in GROUPS.items():
        lines = [title, '='*len(title), '', '.. This file is generated by docs/scripts/generate_api.py.', '']
        if key == 'data':
            lines += ['Processing functions below are exported by ``bruce.data`` unless a',
                      'submodule path is shown. Archive helpers require network access; the',
                      'institutional helpers additionally require a local implementation configured',
                      'with BRUCE_LOCAL_ARCHIVE; it is not included in public distributions.',
                      '``bruce.data.load_NASA_exoplanet_archive(unique_entries=False)`` is a lazy',
                      'alias for the fully documented submodule function below.', '']
        if key == 'sampler':
            lines += ['These names are exported by ``bruce.sampler``. Install the analysis extra.',
                      'Log priors and likelihoods here use the **larger-is-better** sign.',
                      'The helpers do not establish MCMC convergence: check mixing, effective',
                      'sample size and autocorrelation for your scientific application.', '']
        for module, relative, names in modules:
            tree = ast.parse((ROOT/'src/bruce'/relative).read_text())
            lines += [module, '-'*len(module), '']
            lines += [f'`Source <https://github.com/samgill844/bruce/blob/main/src/bruce/{relative}>`__', '']
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
                    lines += entry(module, node)
        if key == 'data':
            lines += ['NGTS FITS helper (optional GUI dependencies)', '-'*len('NGTS FITS helper (optional GUI dependencies)'), '',
                      '.. py:class:: bruce.data.ngtsfit.NGTSFIT_DATA(filename)', '',
                      '   Load a specialised NGTS FITS product. Requires the gui and analysis extras,',
                      '   even for the Matplotlib plotting method, because the module imports Qt.',
                      '   This class is not exported directly by bruce.data.', '',
                      '   :param filename: Path to the NGTS FITS file; constructor reads OPTAB and loads tables.', '']
            tree = ast.parse((ROOT/'src/bruce/data/ngtsfit.py').read_text())
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and not method.name.startswith('_'):
                            lines += entry('bruce.data.ngtsfit.NGTSFIT_DATA', method, 'method')
        (OUT/(key+'.rst')).write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
