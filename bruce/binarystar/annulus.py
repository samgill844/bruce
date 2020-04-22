import numba
import numpy as np 
import matplotlib.pyplot as plt 
import math 
from bruce.binarystar.qpower2 import Flux_drop_analytical_power_2
from .limbdarkening import limb_darkening_intensity

@numba.njit
def area(d, x, R):
	arg1 = (d*d + x*x - R*R)/(2.*d*x)
	arg2 = (d*d + R*R - x*x)/(2.*d*R)
	arg3 = max((-d + x + R)*(d + x - R)*(d - x + R)*(d + x + R), 0.)

	if(x <= (R - d)) :  return math.pi*x*x						
	elif(x >= (R + d)) : return math.pi*R*R				
	else :  return x*x*math.acos(arg1) + R*R*math.acos(arg2) - 0.5*math.sqrt(arg3)

@numba.njit
def Flux_drop_annulus(z, k, ld_law, ldc_1_1, ldc_1_2,ldc_1_3, ldc_1_4, fac = 0.001):
    x_in = max(z - k, 0.)
    x_out = min(z + k, 1.)
    if x_in >= 1. : return 1. 
    elif ((x_out - x_in) < 1e-7) : return 1.
    else:
        delta = 0.
        x = x_in
        dx = fac*math.acos(x)
        
        x += dx  
        A_i = 0. 

        while (x < x_out):
            A_f = area(z, x, k)
            I = limb_darkening_intensity(ld_law, ldc_1_1, ldc_1_2,ldc_1_3, ldc_1_4,   (x - dx/2) ) 
            delta += (A_f - A_i)*I
            dx = fac*math.acos(x)
            x = x + dx
            A_i = A_f


        dx = x_out - x + dx;  
        x = x_out;				
        A_f = area(z, x, k)				
        I = limb_darkening_intensity(ld_law, ldc_1_1, ldc_1_2,ldc_1_3, ldc_1_4,   (x - dx/2) ) 	
        delta += (A_f - A_i)*I	
        return - delta





def Annulus_calc_err(fac, t = np.linspace(-0.05,0.05,1000), fac_lo = 5.0e-4, ld_law_1=2, ldc_1_1=0.8, ldc_1_2=0.8,ldc_1_3=0.8, ldc_1_4=0.8, plot=True ):
    from bruce.binarystar import lc
    F0 = lc(t, integration_type=1, Annulus_fac = fac_lo,   ld_law_1=ld_law_1, ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_2)
    F = lc(t, integration_type=1, Annulus_fac = fac,       ld_law_1=ld_law_1, ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_2)

    if plot:
        f = plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(t, 1e6*(1 - F/F0), 'k')
        ax2.plot(t, F0, 'k', alpha = 0.4)

        ax1.set_ylabel('Noise [ppm]')
        ax2.set_ylabel('Flux')
        ax1.set_xlabel('Time [d]')
        ax1.set_title('Max noise = {:.0f} ppm'.format(np.max(1e6*(1 - F/F0))))
        f.tight_layout()
        return f, np.max(1e6*(1 - F/F0))
    else: return np.max(1e6*(1 - F/F0))


def Annulus_get_fac(err_ppm, t = np.linspace(-0.05,0.05,1000), fac_lo = 5.0e-4,fac_hi=1., ld_law_1=2, ldc_1_1=0.8, ldc_1_2=0.8,ldc_1_3=0.8, ldc_1_4=0.8, verbose=True ):
    from bruce.binarystar import lc
    F0 = lc(t, integration_type=1, Annulus_fac = fac_lo,   ld_law_1=ld_law_1, ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_2)
    err = np.inf 
    n = 0
    if verbose : print('{:>15} | {:>15} | {:>15}'.format('Iter', 'Err [ppm]', 'Diff [ppm]'))
    while  (err > err_ppm) or (err < 0.999*err_ppm):
        fac = (fac_lo + fac_hi)/2.
        F = lc(t, integration_type=1, Annulus_fac = fac,       ld_law_1=ld_law_1, ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_2)
        err = np.max(1e6*(1 - F/F0))
        n+=1
        if err > err_ppm : fac_hi = fac
        else : fac_lo = fac 
        if verbose : print('{:>15.0f} | {:>15.0f} | {:>15.0f}'.format(n, err, err-err_ppm))

        if n > 100 : raise ValueError('Too many iterations')
    return fac
