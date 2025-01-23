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
	
	# Get the width
	width = transit_width(radius_1, k, np.cos(incl)/radius_1, period=period)

	# Check the time steps
	if time_trial is None:
		if time_step is None : time_step = width / 20.
		time_trial = np.arange(np.min(time) - width/2., np.max(time)+width/2., time_step)
		time_trial_mask = check_proximity_of_timestamps(time_trial, time, width)
		time_trial = time_trial[time_trial_mask
						  ]
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
