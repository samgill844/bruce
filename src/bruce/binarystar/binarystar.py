import bruce_c, numpy as np

def lc(t = np.linspace(0,1,100),flux=None, flux_err=None,
        t_zero=0., period = 1.,
        radius_1=0.2, k = 0.2, incl=np.pi/2,
        e=0., w = np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = 2,
        accurate_tp=0,
        jitter=0., offset=1):
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