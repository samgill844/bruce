import bruce_c, numpy as np

from astropy import constants as c

mean_solar_day = 86400.002
R_SunN    = 6.957E8           # m, solar radius 
GM_SunN   = 1.3271244E20      # m3.s-2, solar mass parameter
V_SunN    = 4*np.pi*R_SunN**3/3  # m3,  solar volume 
_arsun   = (GM_SunN*mean_solar_day**2/(4*np.pi**2))**(1/3.)/R_SunN
_rhostar = 3*np.pi*V_SunN/(GM_SunN*mean_solar_day**2)
R = c.R.value

def transit_width(radius_1, k, b, period=1.):
    """
    Total ciurcular transit duration.
    See equation (3) from Seager and Malen-Ornelas, 2003ApJ...585.1038S.
    :param radius_1: R_star/a
    :param k: R_planet/R_star
    :param b: impact parameter = a.cos(i)/R_star
    :param P: orbital period (optional, default P=1)
    :returns: Total transit duration in the same units as P.
    """
    return  period*np.arcsin(radius_1*np.sqrt( ((1+k)**2-b**2) / (1-b**2*radius_1**2) ))/np.pi


def lc(t = np.linspace(-0.2,0.2,100),flux=None, flux_err=None,
        t_zero=0., period = 1.,
        radius_1=0.2, k = 0.2, incl=np.pi/2,
        e=0., w = np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = 2,
        accurate_tp=1,
        jitter=0., offset=0):
    '''
    Calculate the light curve for a given set of parameters. 
    If flux is None, returns the light curve. 
    If flux is not None, returns the log-likelihood given the data
    and model parameters.
    
    Parameters:
        t: array of times
        flux: array of flux values
        flux_err: array of flux errors
        t_zero: time of mid transit
        period: period of transit [days]
        radius_1: scaled radius of star 1 [R1/a]
        k: radius ratio [R2/R1]
        incl: inclination [radians]
        e: eccentricity
        w: argument of periastron passage [radians]
        c: limb darkening coefficient [power-2 law]
        alpha: limb darkening coefficient [power-2 law]
        cadence: cadence [days] [default 0]
        noversample: oversampling factor [default 10]
        light_3: contaminating light (light of other sources/light of target) [default 0]
        ld_law: limb darkening law [default 2 power-2]
        -2: power-2 law [c and alpha] with only transit modelled around t_zero, ignores other cycles
        -1: Box depth given by k, good for BLS and other routines.
            0: no limb darkening
            1: linear limb darkening (from c limb darkening coefficient)
            2: power-2 law [c and alpha]
        accurate_tp: use Newton-Raphson method for Eccentric anomaly [default 1]
        jitter: Jitter Value added in quadrature to flux_err.
        
    Returns:
        if flux is None:
            array of flux values
        else:
            log-likelihood (double)
        '''

    if flux is None:
        return bruce_c.lc(t, 
                t_zero, period, 
                radius_1, k, incl, 
                e, w, 
                c, alpha, 
                cadence, noversample,   
                light_3, 
                ld_law, 
                accurate_tp)
    else:
        return bruce_c.lc_loglike(t, flux, flux_err, 
                t_zero, period, 
                radius_1, k, incl, 
                e, w, 
                c, alpha, 
                cadence, noversample,   
                light_3, 
                ld_law, 
                accurate_tp,
                jitter, offset)
    


def rv1(t = np.linspace(0,1,100),rv=None, rv_err=None,
        t_zero=  0., period = 1.,
        K1 = 1., e = 0.,  w = np.pi / 2.,
        V0 = 0., incl = np.pi / 2.,
        accurate_tp=1,
        jitter=0., offset=0):
    if rv is None:
        return bruce_c.rv1(t,
                           t_zero, period, 
                           K1, e, w,
                           incl,
                           V0,
                           accurate_tp)
    
    else:
        return bruce_c.rv1_loglike(t,rv, rv_err,
                           t_zero, period, 
                           K1, e, w,
                           incl,
                           V0,
                           accurate_tp,
                           offset, jitter) 
    

def rv2(t = np.linspace(0,1,100),rv1=None, rv1_err=None,rv2=None, rv2_err=None,
        t_zero=  0., period = 1.,
        K1 = 1., K2=1.,
        e = 0.,  w = np.pi / 2.,
        V0 = 0., incl = np.pi / 2.,
        accurate_tp=1,
        jitter=0., offset=0):
    if rv1 is None:
        return bruce_c.rv2(t,
                           t_zero, period, 
                           K1,K2,
                           e, w,
                           incl,
                           V0,
                           accurate_tp)    
    else:
        return bruce_c.rv2_loglike(t, rv1, rv2, rv1_err, rv2_err,
                           t_zero, period, 
                           K1,K2,
                           e, w,
                           incl,
                           V0,
                           accurate_tp,
                           offset, jitter) 
    

def stellar_density(radius_1, P, q=0, e = 0., w = np.pi/2.):
    """ 
    Mean stellar density from scaled stellar radius.
    :param radius_1: radius of star in units of the semi-major axis, radius_1 = R_*/a
    :param P: orbital period in mean solar days
    :param q: mass ratio, m_2/m_1
    :returns: Mean stellar density in solar units
    # Eccentricity modification from https://arxiv.org/pdf/1505.02814.pdf   and  https://arxiv.org/pdf/1203.5537.pdf
    """
    fac =  (  ( ((1 - e**2))**1.5   ) / (( 1 + e*np.sin(w) )**3) )  
    return (_rhostar/(P**2*(1+q)*radius_1**3))/fac 