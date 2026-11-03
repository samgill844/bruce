from .data_processing import median_filter, convolve_1d, bin_data, find_nights_from_data, flux_to_mags, mags_to_flux, phase_times, check_proximity_of_timestamps, flatten_data_with_function, bin_data_adaptive

def load_NASA_exoplanet_archive(*args, **kwargs):
    """Load the cached archive, importing network dependencies on demand."""
    from .nasa_archive import load_NASA_exoplanet_archive as load
    return load(*args, **kwargs)
