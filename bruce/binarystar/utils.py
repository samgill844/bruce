

###########################################################################
#                              Handy functions                            #
###########################################################################

def lc_bin(time, flux, bin_width):
        '''
        Function to bin the data into bins of a given width. time and bin_width 
        must have the same units
        '''
        
        edges = np.arange(np.min(time), np.max(time), bin_width)
        dig = np.digitize(time, edges)
        time_binned = (edges[1:] + edges[:-1]) / 2
        flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
        err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else sem(flux[dig == i]) for i in range(1, len(edges))])
        time_bin = time_binned[~np.isnan(err_binned)]
        err_bin = err_binned[~np.isnan(err_binned)]
        flux_bin = flux_binned[~np.isnan(err_binned)]   
        
        return time_bin, flux_bin, err_bin

def transit_width(r, k, b, P=1., fs=0., fc=0., arr=False):
    """
    Total transit duration.
    See equation (3) from Seager and Malen-Ornelas, 2003ApJ...585.1038S.
    :param r: R_star/a
    :param k: R_planet/R_star
    :param b: impact parameter = a.cos(i)/R_star
    :param P: orbital period (optional, default P=1)
    :returns: Total transit duration in the same units as P.
    """
    width = P*np.arcsin(r*np.sqrt( ((1+k)**2-b**2) / (1-b**2*r**2) ))/np.pi
    if arr : width[np.isnan(width) | np.isinf(width)] = 0.05
    else : 
        if np.isnan(width) or np.isinf(width) : width = 0.05
    return width

def rhostar(r_1, P, q=0, fs=0., fc=0.):
    """ 
    Mean stellar density from scaled stellar radius.
    :param r_1: radius of star in units of the semi-major axis, r_1 = R_*/a
    :param P: orbital period in mean solar days
    :param q: mass ratio, m_2/m_1
    :returns: Mean stellar density in solar units
    # Eccentricity modification from https://arxiv.org/pdf/1505.02814.pdf   and  https://arxiv.org/pdf/1203.5537.pdf
    """
    w = math.atan2(fs,fc)
    e = clip(fs**2 + fc**2,0,0.999) 
    fac =  (  ( ((1 - e**2))**1.5   ) / (( 1 + e*np.sin(omega) )**3) )  
    return (_rhostar/(P**2*(1+q)*r_1**3))/fac  

# Phase
def phaser(time, t_zero, period) : return ((time - t_zero)/period) - np.floor((time - t_zero)/period) 

