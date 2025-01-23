import bruce_c, numpy as np


###############################################################
#                  bruce_c wrappers                           #
###############################################################
def median_filter(time, flux, bin_size=0.5/24/3) : return bruce_c.median_filter(time, flux, bin_size)
def convolve_1d(time, flux, bin_size=0.5/24/3) : return bruce_c.convolve_1d(time, flux, bin_size)
def bin_data(time, flux, bin_size=0.5/24/3) : return bruce_c.bin_data(time, flux, bin_size)
def check_proximity_of_timestamps(time_trial, time, width) : return bruce_c.check_proximity_of_timestamps(time_trial, time, width)



###############################################################
#                  Other functions                            #
###############################################################
def find_nights_from_data(x, dx_lim):
    '''
    Split array buy time gaps. Can be used to get individual nights in datasets.
    '''
    dx = np.gradient(x) 
    dx_thresh = np.sort(np.where(dx >= dx_lim)[0] +1)

    # Now check for consecutive integers and delte them
    delete_idxs = []
    i = 0 
    while i < (dx_thresh.shape[0]-1):
        if (dx_thresh[i]+1) == dx_thresh[i+1] :
            delete_idxs.append(i+1)
            i +=2
        else : i +=1
    dx_thresh = np.delete(dx_thresh, delete_idxs)

    # create the idx to split
    idx = np.arange(x.shape[0])
    return np.split(idx, dx_thresh)



def mags_to_flux(mags, mags_err):
    """
    Take 2 arrays, light curve and errors
    and convert them from differential magnitudes
    back to relative fluxes
    Rolling back these equations:
        mags = - 2.5 * log10(flux)
        mag_err = (2.5/log(10))*(flux_err/flux)
    """
    flux = 10.**(mags / -2.5)
    flux_err = (mags_err/(2.5/np.log(10))) * flux
    return flux, flux_err

def flux_to_mags(flux, flux_err):
    """
    Take 2 arrays, light curve and errors
    and convert them from differential magnitudes
    back to relative fluxes
    Applying these equations:
        mags = - 2.5 * log10(flux)
        mag_err = (2.5/log(10))*(flux_err/flux)
    """
    mags = -2.5*np.log10(flux)
    mags_err = (2.5/np.log(10))*(flux_err/flux)
    return mags, mags_err

def phase_times(times, epoch, period, phase_offset=0.0):
    """
    Take a list of times, an epoch, a period and
    convert the times into phase space.
    An optional offset can be supplied to shift phase 0
    """
    return (((times - epoch)/period)+phase_offset)%1 - phase_offset