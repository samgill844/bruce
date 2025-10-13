# Imports 
import numpy as np
import bruce
import matplotlib.pyplot as plt


def get_theta(name,theta,theta_names) : return theta[np.argwhere(theta_names==name)[0][0]]

def chi_fixed_period(theta, data, theta_names, 
                     plot=False, fit_period=30.,
                    median_bin_size = 1,convolve_bin_size = 1):
    # Check 
    if get_theta('b',theta,theta_names) > (1+get_theta('k',theta,theta_names)) : return np.inf 
    
    # Convert
    incl = np.arccos(get_theta('radius_1',theta,theta_names)*get_theta('b',theta,theta_names))
    
    # Get the model
    model = bruce.binarystar.lc(data.time,t_zero=get_theta('t_zero',theta,theta_names), period=fit_period, radius_1=get_theta('radius_1',theta,theta_names), k=get_theta('k',theta,theta_names), incl=incl, ld_law=-2)
    
    # Get the residuals
    residuals = data.flux - model
    
    # Now fflatten it with a convolved median filter
    if median_bin_size is not None:
        w = bruce.data.median_filter(data.time, residuals, bin_size=median_bin_size)
        w = bruce.data.convolve_1d(data.time, w, bin_size=convolve_bin_size)
    else : 
        w = data.w.copy()
    
    if plot:
        fig, ax = plt.subplots(1,2, gridspec_kw={'hspace' : 0, 'wspace' : 0})
        ax[0].errorbar(data.time-get_theta('t_zero',theta,theta_names), data.flux, yerr=data.flux_err, fmt='k.', alpha = 0.1)
        ax[0].plot(data.time-get_theta('t_zero',theta,theta_names), model * (w), c='orange')
        ax[0].plot(data.time-get_theta('t_zero',theta,theta_names), (w), c='orange', ls='--')
        #ax[0].set(xlabel='Time from Transit [d]', ylabel='Flux')
        
        ax[1].errorbar(data.time-get_theta('t_zero',theta,theta_names), data.flux/(w), yerr=data.flux_err/(w), fmt='k.', alpha = 0.1)
        ax[1].plot(data.time-get_theta('t_zero',theta,theta_names), model, c='orange')
        ax[1].set(yticks=[])

        fig.supxlabel('Time from Transit [d]', fontsize=18, x=0.55)
        fig.supylabel('Flux', fontsize=18)
                    
        #width = bruce.binarystar.transit_width(get_theta('radius_1',theta,theta_names), get_theta('k',theta,theta_names), get_theta('b',theta,theta_names), period=fit_period)
        #ax[1].set_xlim(get_theta('t_zero',theta,theta_names)-width, get_theta('t_zero',theta,theta_names)+width)

        return_data = (data.time-get_theta('t_zero',theta,theta_names), data.flux/(w), data.flux_err/(w),
                       data.time-get_theta('t_zero',theta,theta_names), model)

        return fig, ax, return_data
    
    # Get the chi-squared
    return -2*bruce.sampler.loglike(data.flux, data.flux_err, model * (w), jitter=0., offset=False)
