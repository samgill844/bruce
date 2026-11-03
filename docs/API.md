# Python API catalogue

Generated from the source definitions. Import paths are shown explicitly; some definitions are available only from their submodule. Signatures and docstrings in the linked source remain authoritative. Private helpers and nested functions are omitted.

## `bruce.ambiguous_period.ambiguous_period`

[Source](../src/bruce/ambiguous_period/ambiguous_period.py)

- `ambiguous_period` (class)
  - `ambiguous_period.group_data_by_epochs`
  - `ambiguous_period.mask_and_filter_events`
  - `ambiguous_period.are_multiples_within_tolerance`
  - `ambiguous_period.calcualte_aliases`
  - `ambiguous_period.calcualte_data_delta_L`
  - `ambiguous_period.plot_aliases`
  - `ambiguous_period.calculate_alias_probability_period`
  - `ambiguous_period.stellar_density`
  - `ambiguous_period.calcualte_alias_probability_density`
  - `ambiguous_period.calculate_probability_with_data`
  - `ambiguous_period.calcualate_alias_probabilities`
  - `ambiguous_period.count_delta_L_solutions`
  - `ambiguous_period.evaluate_peaks`
  - `ambiguous_period.plot_solution`
  - `ambiguous_period.transit_plan`
  - `ambiguous_period.plot_all_events`
  - `ambiguous_period.plot_event`
  - `ambiguous_period.calcualte_viable_aliases_from_delta_L`

## `bruce.ambiguous_period.data_retrieval`

[Source](../src/bruce/ambiguous_period/data_retrieval.py)

- `extract_zip`
- `safe_extract`
- `download_tess_data`
- `get_tic8`
- `get_gaia_dr2`
- `get_gaia_dr3`

## `bruce.ambiguous_period.fitting`

[Source](../src/bruce/ambiguous_period/fitting.py)

- `get_theta`
- `chi_fixed_period`

## `bruce.ambiguous_period.mono_event`

[Source](../src/bruce/ambiguous_period/mono_event.py)

- `group_data_by_epochs`
- `photometry_time_series` (class)
  - `photometry_time_series.plot_segments`
  - `photometry_time_series.flatten_data`
  - `photometry_time_series.flatten_data_old`
  - `photometry_time_series.fit_transit_event`
  - `photometry_time_series.write_data`
  - `photometry_time_series.load_data`
  - `photometry_time_series.normalise_gui`
  - `photometry_time_series.split_and_normalise_transit_events`
- `get_theta`
- `get_de`
- `report_best_params`
- `mono_event` (class)
  - `mono_event.de_transit_width`
  - `mono_event.de_get_epoch`
  - `mono_event.de_get_radius_1`
  - `mono_event.de_get_k`
  - `mono_event.de_get_b`
  - `mono_event.de_evaluate_model`
  - `mono_event.de_phase_model`
  - `mono_event.fit_event_with_fixed_period`

## `bruce.binarystar.binarystar`

[Source](../src/bruce/binarystar/binarystar.py)

- `transit_width`
- `lc`
- `rv1`
- `rv2`
- `stellar_density`

## `bruce.binarystar.ttv`

[Source](../src/bruce/binarystar/ttv.py)

- `chopping_model`
- `residual`
- `scan_companion_period`
- `nearest_resonances`
- `synodic_period`
- `synodic_to_periods`
- `branch_geometry`
- `inspect_chopping_branches`
- `period_to_a`
- `plot_single_epoch`
- `synodic_phase`
- `stroboscopic_ttv_plot`
- `plot_period_scan`
- `plot_ttv_model`

## `bruce.data.data_processing`

[Source](../src/bruce/data/data_processing.py)

- `median_filter`
- `convolve_1d`
- `bin_data`
- `bin_data_adaptive`
- `check_proximity_of_timestamps`
- `find_optimal_ngts_filter_window`
- `find_nights_from_data`
- `mags_to_flux`
- `flux_to_mags`
- `phase_times`
- `flatten_data_with_function`

## `bruce.data.nasa_archive`

[Source](../src/bruce/data/nasa_archive.py)

- `download_NASA_exoplanet_archive`
- `query_simbad`
- `load_NASA_exoplanet_archive`
- `download_tois_ctois`
- `load_toi_table`
- `load_ctoi_table`
- `load_tic8_info_for_target`
- `load_gaia_table`

## `bruce.data.ngtsfit`

[Source](../src/bruce/data/ngtsfit.py)

- `NGTSFIT_DATA` (class)
  - `NGTSFIT_DATA.load_data`
  - `NGTSFIT_DATA.plot_action_summary_matplotlib`
  - `NGTSFIT_DATA.plot_action_summary_pyqtgraph`

## `bruce.empirical_relations.empirical_relations`

[Source](../src/bruce/empirical_relations/empirical_relations.py)

- `mass_radius`
- `radius_to_mass`

## `bruce.sampler.sampler`

[Source](../src/bruce/sampler/sampler.py)

- `prior_uniform`
- `prior_normal`
- `prior_bounded_normal`
- `loglike_`
- `loglike`
- `fsfc_to_ew`
- `ew_to_fsfc`
- `ca_to_h1h2`
- `h1h2_to_ca`
- `incl_from_radius_1_b`
- `get_sampler_report`
- `bic`
- `create_starting_positions`
- `intilise_and_run_sampler`
- `analyse_sampler`
- `corner_sampler`
- `corner_sampler_blobs`
- `get_assymetric_latex_format`
- `print_fmt_latex`
- `print_vals_from_sampler_latex`

## `bruce.template_match.template_match`

[Source](../src/bruce/template_match/template_match.py)

- `template_match_lightcurve`
- `get_delta_loglike_height_from_fap`
- `phase_disperison`
- `autoperiod`
- `template_match_snr`

## `bruce.visual.animate_orbits`

[Source](../src/bruce/visual/animate_orbits.py)

- `animate_orbit`
