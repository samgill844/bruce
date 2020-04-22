# Standard imports
import numba, numba.cuda, numpy as np , math 
from numba import prange 

# bruce imports
from .kepler import getTrueAnomaly, get_z, getProjectedPosition
from .eker_spots import eker_spots
from .doppler import Fdoppler
from .ellipsoidal import Fellipsoidal 
from .quadratic import Flux_drop_analytical_quadratic 
from .qpower2 import Flux_drop_analytical_power_2
from .uniform import Flux_drop_uniform , area
from .reflection import reflection
from .annulus import Flux_drop_annulus
from time import time as _time


@numba.njit
def clip(a, b, c):
    if (a < b)      :  return b
    elif (a > c)    :  return c
    else            :  return a


@numba.njit
def _lc_engine(time, zp,
        t_zero, period,
        radius_1, k ,
        e,w,sini,
        nspots, nflares,
        q, albedo,
        alpha_doppler, K1,
        spots, flares, omega_1, 
        incl,
        ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
        SBR, light_3,
        A_g,xyz,
        E_tol,
        integration_type, Annulus_fac ):

       # Get true anomaly
        if e ==0 : nu = ((time - t_zero)/period - math.floor(((time - t_zero)/period)))*2.*math.pi + math.pi/2
        else : nu = getTrueAnomaly(time, e, w, period,  t_zero, incl, E_tol, radius_1 ) 

        # Get projected orbital seperation
        z = get_z(nu, e, incl, w, radius_1)  
        
        # Initialse the flux
        # The model we will use is:
        #   F_tot = continuum + F_ecl + F_ellipsoidal + F_spots + F_transit 
        #   F_ellipsoidal -> ellipsoidal effect contralled by the mass ratio, q and the gravity limb-darkening coefficient
        #   F_spots       -> flux drop from spots controlled by the eker model
        #   F_transit     -> flux from the desired transit model (using 1 as the continuum)

        continuum = 1.0 # continuum at 1.0 (normalised)
        F_spots = 0.    # spot flux 
        F_flares = 0.    # flare flux 
        F_doppler = 0.  # Doppler beaming
        F_ellipsoidal = 0.  # Ellipsoidal variation
        F_reflection = 0.0  # reflection
        F_transit =  0.0    # transit flux drop

        # First, let's check for spots 
        if (nspots > 0):
            spot_phase = omega_1*2*math.pi*(time - t_zero)/period
            for j in range(nspots) : F_spots += eker_spots(spots[j*4 + 0], spots[j*4 +1], incl, spots[j*4 +2], spots[j*4 +3], 0.5,0.3, spot_phase)

        # Now let's check for flares
        # from Eqn. 1 from https://academic.oup.com/mnras/article/445/3/2268/2907951
        if (nflares > 0):
            for j in range(nflares):
                if time <= flares[j*4] : F_flares += flares[j*4+1]*math.exp(-(time - flares[j*4])**2 / (2*flares[j*4+2]**2))
                else                     : F_flares += flares[j*4+1]*math.exp(-(time - flares[j*4]) / (2*flares[j*4+3]))

        # Next, we need to check for doppler beaming
        # Check for doppler beaming 
        if (alpha_doppler > 0) and (K1 > 0) : F_doppler = Fdoppler(nu, alpha_doppler, K1 )

        # Check for eelipsoidal variation and apply it if needed
        if (q>0.):
            alpha = math.acos(math.sin(w + nu)*math.sin(incl))
            F_ellipsoidal = Fellipsoidal(alpha, q, radius_1, incl, 0.5, gdc_1)

        # Next, reflection effect
        if (A_g > 0) : F_reflection = reflection(time, t_zero, period, A_g, radius_1, e, w, sini, xyz,)

        # Check distance between them to see if its transiting
        if (z < (1.0+ k)):
            # So it's eclipsing, lets find out if its a primary or secondary
            f = getProjectedPosition(nu, w , incl)

            if (f > 0):
                # First, get the flux drop
                if (integration_type==0):
                    if (ld_law_1==0) : F_transit = Flux_drop_uniform( z, k, 1.) # uniform limb-darkening 
                    if (ld_law_1==1) : F_transit = Flux_drop_analytical_quadratic( z,  k,  ldc_1_1,  ldc_1_2,  1e-8)
                    if (ld_law_1==2) : F_transit = Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_2, 1E-8) 
                    if (ld_law_1==-1) : F_transit = -1.
                    elif ((ld_law_1==-2) and (abs((time - t_zero) / period) < 0.5)) : F_transit = Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_2, 1E-8) 
                elif (integration_type==1):
                    F_transit = Flux_drop_annulus(z, k, ld_law_1, ldc_1_1, ldc_1_2,0., 0., Annulus_fac)

                # Now account for SBR (third light)
                if (SBR>0) : F_transit = F_transit*(1. - k*k*SBR) 

            elif (SBR>0.) :
                if (ld_law_1!=-1) : F_transit =  Flux_drop_uniform(z, k, SBR) # Secondary eclipse
                else : F_transit = -1.


        # Now put the model together
        model = continuum + F_spots + F_flares + F_doppler + F_ellipsoidal + F_reflection + F_transit

        # That's all from the star, so let's account for third light 
        if (light_3 > 0.0) : model = model/(1. + light_3) + (1.-1.0/(1. + light_3)) # third light

        # Now return model
        return model

@numba.njit
def _lc(time, LC, LC_ERR, J, zp,
        t_zero, period,
        radius_1, k ,
        fs, fc, dw,
        q, albedo,
        alpha_doppler, K1,
        spots, flares, omega_1, 
        incl,
        ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
        SBR, light_3,
        cadence, noversample, 
        A_g,xyz,
        E_tol,
        integration_type, Annulus_fac,
        loglike_switch ):

    # Unpack and convert (assume incl in radians)
    w = math.atan2(fs, fc) 
    e = clip(fs**2 + fc**2,0,0.999) 
    sini = math.sin(incl*math.pi/180.)
    nspots = spots.shape[0]//4 
    nflares = flares.shape[0]//4 

    loglike=0.
    for i in range(time.shape[0]):
        if cadence != -1.:
            dr = (cadence/2) / ((noversample-1)/2)
            model = 0.
            for j in range(noversample) : model += _lc_engine(time[i] -dr*((noversample-1)/2) + j*dr , zp,
                                                        t_zero, period,
                                                        radius_1, k ,
                                                        e,w,sini,
                                                        nspots, nflares,
                                                        q, albedo,
                                                        alpha_doppler, K1,
                                                        spots, flares, omega_1, 
                                                        incl,
                                                        ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
                                                        SBR, light_3,
                                                        A_g,xyz,
                                                        E_tol,
                                                        integration_type, Annulus_fac )    
            model /= noversample
        else:
            model = _lc_engine(time[i], zp,
                    t_zero, period,
                    radius_1, k ,
                    e,w,sini,
                    nspots, nflares,
                    q, albedo,
                    alpha_doppler, K1,
                    spots, flares, omega_1, 
                    incl,
                    ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
                    SBR, light_3,
                    A_g,xyz,
                    E_tol ,
                    integration_type, Annulus_fac)

        if loglike_switch : 
            model = zp - 2.5*math.log10(model)
            wt = 1.0 / (LC_ERR[i]**2 + J**2)
            loglike += -0.5*((LC[i] - model)**2*wt) #- math.log(wt))
        else : LC[i] = model 

    return loglike 


@numba.cuda.jit
def kernel_lc(time, LC, LC_ERR, J, zp,
        t_zero, period,
        radius_1, k ,
        fs, fc, dw,
        q, albedo,
        alpha_doppler, K1,
        spots, flares, omega_1, 
        incl,
        ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
        SBR, light_3,
        cadence, noversample, 
        A_g,xyz,
        E_tol,
        integration_type, Annulus_fac,
        loglike ):


    # Unpack and convert (assume incl in radians)
    # Unpack and convert (assume incl in radians)
    w = math.atan2(fs, fc) 
    e = clip(fs**2 + fc**2,0,0.999) 
    sini = math.sin(incl*math.pi/180.)
    nspots = spots.shape[0]//4 
    nflares = flares.shape[0]//4 
    i = numba.cuda.grid(1)

    if cadence != -1.:
        dr = (cadence/2) / ((noversample-1)/2)
        model = 0.
        for j in range(noversample) : model += _lc_engine(time[i] -dr*((noversample-1)/2) + j*dr , zp,
                                                    t_zero, period,
                                                    radius_1, k ,
                                                    e,w,sini,
                                                    nspots, nflares,
                                                    q, albedo,
                                                    alpha_doppler, K1,
                                                    spots, flares, omega_1, 
                                                    incl,
                                                    ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
                                                    SBR, light_3,
                                                    A_g,xyz,
                                                    E_tol ,
                                                    integration_type, Annulus_fac )    
        model /= noversample
    else:
        model = _lc_engine(time[i], zp,
                t_zero, period,
                radius_1, k ,
                e,w,sini,
                nspots, nflares,
                q, albedo,
                alpha_doppler, K1,
                spots, flares, omega_1, 
                incl,
                ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
                SBR, light_3,
                A_g,xyz,
                E_tol ,
                integration_type, Annulus_fac )    


    model = zp - 2.5*math.log10(model)
    wt = 1.0 / (LC_ERR[i]**2 + J**2)
    #loglike[i] = -0.5*((LC[i] - model)**2*wt)
    
    # Numba atomic add
    numba.cuda.atomic.add(loglike, 0, -0.5*((LC[i] - model)**2*wt) )

@numba.cuda.reduce
def sum_reduce(a, b):
    return a + b




def lc(time, LC=np.zeros(1), LC_ERR=np.zeros(1), J=0., zp=0.,
    t_zero=0., period=1.,
    radius_1=0.2, k=0.2 ,
    fs=0., fc=0., dw = 0.,
    q=0., albedo=0.,
    alpha_doppler=0., K1=0.,
    spots = np.zeros(1),flares = np.zeros(1), omega_1=1., 
    incl = 90.,
    ld_law_1=2, ldc_1_1=0.8, ldc_1_2=0.8, gdc_1=0.4,
    SBR=0., light_3=0.,
    A_g = 0.,xyz = np.zeros(3, dtype = np.float32),
    E_tol=1e-5, 
    cadence = -1., noversample = 9, 
    gpu=0, loglike=np.zeros(1, dtype = np.float64), blocks = -1, threads_per_block = -1,
    integration_type = 0, Annulus_fac = 0.001):

    incl = np.pi * incl/180.

    spots = np.ravel(spots).astype(np.float64)
    flares = np.ravel(flares).astype(np.float64)

    if not gpu:
        # First, let's see if we need loglike or not!
        if LC_ERR[0]==0 : loglike_switch = 0
        else            : loglike_switch = 1

        # Now, let's initiase the arrays, if needed
        if not loglike_switch : LC = np.empty_like(time) 

        # Now make the call
        loglike = _lc(time, LC, LC_ERR, J, zp,
            t_zero, period,
            radius_1, k ,
            fs, fc, dw,
            q, albedo,
            alpha_doppler, K1,
            spots, flares, omega_1, 
            incl,
            ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
            SBR, light_3,
            cadence, noversample, 
            A_g,xyz,
            E_tol,
            integration_type, Annulus_fac,
            loglike_switch )

        if loglike_switch : return loglike 
        else              : return LC 

    if gpu:
        # Loglike ony supported 

        # First, lets copy the loglike array for the atomic add
        loglike = numba.cuda.to_device(loglike)

        # Now lets caluculate threads_per_block and blocks if not passed
        if threads_per_block == -1 :  threads_per_block = 256
        if blocks == -1 : blocks = int(np.ceil(time.shape[0]/threads_per_block))

        ## Call the kernel to populate loglike
        #start = _time() 
        kernel_lc[blocks, threads_per_block](time, LC, LC_ERR, J, zp,
            t_zero, period,
            radius_1, k ,
            fs, fc, dw,
            q, albedo,
            alpha_doppler, K1,
            spots, flares, omega_1, 
            incl,
            ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
            SBR, light_3,
            cadence, noversample, 
            A_g,xyz,
            E_tol,
            integration_type, Annulus_fac,
            loglike )

        return loglike.copy_to_host()[0]