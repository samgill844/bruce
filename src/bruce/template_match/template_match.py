import numpy as np
from bruce.binarystar import transit_width
from bruce.data import check_proximity_of_timestamps
import bruce_c
import scipy.stats as stats

def template_match_lightcurve(time, flux, flux_err, normalisation_model, period = 1.,
        radius_1=0.2, k = 0.2, incl=np.pi/2,
        e=0., w = np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = -2,
        accurate_tp=1,
		jitter=0., offset=0,
		time_step=None, time_trial=None):
	# Data
	time, flux, flux_err = time.astype(np.float64), flux.astype(np.float64), flux_err.astype(np.float64)
	
	if isinstance(radius_1, np.ndarray):
		width = 0.5 # Fix this
		if time_trial is None:
			if time_step is None : time_step = width / 20.
			time_trial = np.arange(np.min(time) - width/2., np.max(time)+width/2., time_step)
			time_trial_mask = check_proximity_of_timestamps(time_trial, time, width)
			time_trial = time_trial[time_trial_mask]
		# Call
		DeltaL = bruce_c.template_match_batch_reduce(time_trial,
			time, flux, flux_err, normalisation_model, 
			period,
			radius_1, k, incl,
			e, w, 
			c, alpha,
			cadence, noversample,
			light_3,
			ld_law,
			accurate_tp,
			jitter, offset)
	else:
		# Get the width
		width = transit_width(radius_1, k, np.cos(incl)/radius_1, period=period)

		# Check the time steps
		if time_trial is None:
			if time_step is None : time_step = width / 20.
			time_trial = np.arange(np.min(time) - width/2., np.max(time)+width/2., time_step)
			time_trial_mask = check_proximity_of_timestamps(time_trial, time, width)
			time_trial = time_trial[time_trial_mask]
		# Call
		DeltaL = bruce_c.template_match_reduce(time_trial,
			time, flux, flux_err, normalisation_model, 
			width,
			period,
			radius_1, k, incl,
			e, w, 
			c, alpha,
			cadence, noversample,
			light_3,
			ld_law,
			accurate_tp,
			jitter, offset)

	return time_trial, DeltaL


def get_delta_loglike_height_from_fap(p_value=[0.01,0.001,0.0001], df=6):
    # Compute the CDF 
    return p_value, stats.chi2.ppf(1 - np.array(p_value), df)

def phase_disperison(time_trial, peaks, time, flux, flux_err,  
					 	periods=None,
						samples_per_peak=5,
						nyquist_factor=5,
						minimum_period=None,
						maximum_period=None):
    if periods is None : periods = autoperiod(time_trial, samples_per_peak=samples_per_peak, nyquist_factor=nyquist_factor,minimum_period=minimum_period,maximum_period=maximum_period)
    dispersion ,chi_squared = bruce_c.phase_dispersion(time_trial, np.array(peaks, dtype=np.int32),  periods, time, flux, flux_err)
    return periods, dispersion ,chi_squared

def autoperiod( x,
    samples_per_peak=5,
    nyquist_factor=5,
    minimum_period=None,
    maximum_period=None,
    return_freq_limits=False,
):
    """Determine a suitable frequency grid for data.

    Note that this assumes the peak width is driven by the observational
    baseline, which is generally a good assumption when the baseline is
    much larger than the oscillation period.
    If you are searching for periods longer than the baseline of your
    observations, this may not perform well.

    Even with a large baseline, be aware that the maximum frequency
    returned is based on the concept of "average Nyquist frequency", which
    may not be useful for irregularly-sampled data. The maximum frequency
    can be adjusted via the nyquist_factor argument, or through the
    maximum_frequency argument.

    Parameters
    ----------
    samples_per_peak : float, optional
        The approximate number of desired samples across the typical peak
    nyquist_factor : float, optional
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if maximum_frequency is not provided.
    minimum_frequency : float, optional
        If specified, then use this minimum frequency rather than one
        chosen based on the size of the baseline.
    maximum_frequency : float, optional
        If specified, then use this maximum frequency rather than one
        chosen based on the average nyquist frequency.
    return_freq_limits : bool, optional
        if True, return only the frequency limits rather than the full
        frequency grid.

    Returns
    -------
    frequency : ndarray or `~astropy.units.Quantity` ['frequency']
        The heuristically-determined optimal frequency bin
    """
    baseline = x.max() - x.min()
    n_samples = x.size

    df = 1.0 / baseline / samples_per_peak

    if maximum_period is None : minimum_frequency = 0.5 * df
    else : minimum_frequency = 1 / maximum_period

    if minimum_period is None:
        avg_nyquist = 0.5 * n_samples / baseline
        maximum_frequency = nyquist_factor * avg_nyquist
    else : maximum_frequency = 1 / minimum_period

    Nf = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))

    if return_freq_limits:
        return minimum_frequency, minimum_frequency + df * (Nf - 1)
    else:
        return 1/(minimum_frequency + df * np.arange(Nf))