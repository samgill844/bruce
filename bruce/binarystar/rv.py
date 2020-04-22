

# Python imports
import numba, numba.cuda, numpy as np , math 

import matplotlib.pyplot as plt 

# bruce imports
from .kepler import getTrueAnomaly


'''
    *Fit parameters*:
     - epsilon - linear limb dark
     - gamma   - Rp/Rs (ratio of planetary and stellar radius)
     - P       - Orbital period [d]
     - T0      - Central transit time
     - i       - Inclination of orbit [rad]
     - Is      - Inclination of stellar rotation axis [rad]
     - Omega   - Angular rotation velocity (star) [rad/s]
     - lambda  - Sky-projected angle between stellar
       rotation axis and normal of orbit plane [rad]
     - a       - Semi major axis [stellar radii]
'''
@numba.njit(fastmath=True)
def _Xp(nu, lambda_1, incl, a) : 
    result = -math.cos(lambda_1)*math.sin(nu) - math.sin(lambda_1)*math.cos(incl)*math.cos(nu)
    result *= a
    return result

@numba.njit(fastmath=True)
def _Zp(nu, lambda_1, incl, a) : 
    result = math.sin(lambda_1)*math.sin(nu) - math.cos(lambda_1)*math.cos(incl)*math.cos(nu)
    result *= a
    return result

@numba.njit(fastmath=True)
def _g(x, e, g, x0):
    result = (1.0-x**2) * math.asin(math.sqrt((g**2-(x-1.0-e)**2)/(1.0-x**2))) + \
       math.sqrt(2.0*(1.0+e)*(x0-x)*(g**2-(x-1.0-e)**2))
    return result


@numba.njit(fastmath=True)
def _W1( rho, k):
    result = math.sqrt(1.0 - rho**2) - k**2 * (2.0-rho**2)/(8.0*(1.0-rho**2)**(3.0/2.0))
    return result
  
@numba.njit(fastmath=True)
def _W2( rho, k):
    result =  math.sqrt(1.0 - rho**2) - k**2 * (4.0-3.0*rho**2)/(8.0*(1.0-rho**2)**(3.0/2.0))
    return result

@numba.njit(fastmath=True)
def _W3( x0, zeta, xc, etap, k):
    result = math.pi/6.0*(1.0-x0)**2*(2.0+x0) + \
      math.pi/2.0*k*(k-zeta) * \
      _g(xc, etap,k, x0) / _g((1.0-k), -k, k, x0) * \
      _W1(1.0-k, k)
    return result 

@numba.njit(fastmath=True)
def _W4( x0, zeta, xc, etap, k):
    result = math.pi/8.*(1.0-x0)**2*(1.0+x0)**2 + \
      math.pi/2.*k*(k-zeta)*xc * \
      _g(xc, etap, k, x0) / _g((1.0-k), -k, k, x0) * \
      _W2(1.0-k, k)
    return result

@numba.njit
def clip(a, b, c):
    if (a < b)      :  return b
    elif (a > c)    :  return c
    else            :  return a


@numba.njit(fastmath=True)
def _rv( time, RV, RV_err, J,
        t_zero, period, 
        K1, fs, fc, dw,
        V0, dV0,
        incl,
        E_tol,
        v_s, v_c, k, radius_1, q, incl_rot, u_lin_ld,
        loglike_switch):

    # Unpack and convert (assume incl in radians)
    w = math.atan2(fs, fc) 
    e = clip(fs**2 + fc**2,0,0.999) 

    loglike=0.
    for i in range(time.shape[0]):
        nu = getTrueAnomaly(time[i], e, w , period,  t_zero, incl, E_tol, radius_1 ) 
        model = K1*(e*math.cos(w) + math.cos(nu + w)) + V0 + dV0*(time[i] - t_zero) 
        
        if v_s !=0 and v_c != 0 : 
            vsini_1  = v_s**2 + v_c**2
            lambda_1 = math.atan2(v_s,v_c)
            a = 0.019771142*K1*math.sqrt(1-e**2)*(1+1/q)*period/math.sin(incl)
            omega_1 = math.sqrt(vsini_1**2 / 695510**2)
            print(a, omega_1, vsini_1, lambda_1)

            Xp = _Xp(nu, lambda_1, incl, a)
            Zp = _Zp(nu, lambda_1, incl, a) 
            rho = math.sqrt(Xp**2 + Zp**2) 
            etap = math.sqrt(Xp**2 + Zp**2) - 1.0 
            zeta = (2.0*etap + k**2 + etap**2) / (2.0*(1.0+etap)) 
            x0 = 1.0-(k**2 - etap**2)/(2.0*(1.0+etap))
            xc = x0+(zeta-k)/2.0 

            dphase = abs((time[i] - t_zero)/period) 
            dphase = min( dphase-math.floor(dphase), abs(dphase-math.floor(dphase)-1))

            if (rho < (1.0 - k)) and (dphase < 0.25):
                print('Here1')

                W1 = _W1( rho, k)
                W2 = _W2( rho, k)
                model += Xp*omega_1*math.sin(incl_rot)*k**2 * \
                    (1.0 - u_lin_ld*(1.0 - W2)) / \
                    (1.0 - k**2 - u_lin_ld*(1./3. - k**2*(1.0-W1))) 

            if (rho >= 1.-k) and (rho < 1.0 + k) and (dphase < 0.25):
                print('Here2')
                z0 = math.sqrt((k**2 - etap**2)*( (etap+2.0)**2-k**2) ) / (2.0*(1.0+etap))
                W3 = _W3( x0, zeta, xc, etap, k)
                W4 = _W4( x0, zeta, xc, etap, k)

                model += (Xp*omega_1*math.sin(incl_rot)*( \
                    (1.0-u_lin_ld) * (-z0*zeta + k**2*math.acos(zeta/k)) + \
                    (u_lin_ld/(1.0+etap))*W4)) / \
                    (math.pi*(1.-1.0/3.0*u_lin_ld) - (1.0-u_lin_ld) * (math.asin(z0)-(1.+etap)*z0 + \
                    k**2*math.acos(zeta/k)) - u_lin_ld*W3)
            

        if loglike_switch : 
            wt = 1.0 / (RV_err[i]**2 + J**2)
            loglike += -0.5*((RV[i] - model)**2*wt) #- math.log(wt))
        else : RV[i] = model 

    return loglike






@numba.cuda.jit
def kernel_rv(time, RV, RV_err, J,
        t_zero, period, 
        K1, fs, fc, dw,
        V0, dV0,
        incl,
        E_tol, 
        loglike):

    # Unpack and convert (assume incl in radians)
    w = math.atan2(fs, fc)
    e = fs**2 + fc**2

    i = numba.cuda.grid(1)

    nu = getTrueAnomaly(time[i], e, w , period,  t_zero, incl, E_tol ) 
    model = K1*(e*math.cos(w) + math.cos(nu + w )) + V0 + dV0*(time[i] - t_zero) 
    wt = 1.0 / (RV_err[i]**2 + J**2)
    loglike[i] = -0.5*((RV[i] - model)**2*wt - math.log(wt))        


@numba.cuda.reduce
def sum_reduce(a, b):
    return a + b


def rv( time, RV = np.zeros(1), RV_err=np.zeros(1), J=0,
        t_zero=0., period=1., 
        K1=10., fs=0., fc=0., dw = 0.,
        V0=10., dV0=0.,
        incl=90.,
        E_tol=1e-5, 
        v_s=0., v_c=0., k=0.2, radius_1=0.2, q=0., incl_rot=0., u_lin_ld=0.5,
        gpu=0, loglike=np.zeros(1), blocks = 10, threads_per_block = 512):




    # Convert inclination 
    incl = np.pi * incl/180.
    incl_rot = np.pi * incl_rot/180.

    if not gpu:
        # First, let's see if we need loglike or not!
        if RV_err[0]==0 : loglike_switch = 0
        else            : loglike_switch = 1

        # Now, let's initiase the arrays, if needed
        if not loglike_switch : RV = np.empty_like(time) 

        # Now make the call
        loglike = _rv( time, RV, RV_err, J,
            t_zero, period, 
            K1, fs, fc, dw,
            V0, dV0,
            incl,
            E_tol, 
            v_s, v_c, k, radius_1, q, incl_rot, u_lin_ld,
            loglike_switch) 

        if loglike_switch : return loglike 
        else              : return RV 
    
    '''
    if gpu:
        # Loglike ony supported 
        # assumeing loglike is array

        ## Call the kernel to populate loglike 
        kernel_rv[blocks, threads_per_block](time, RV, RV_err, J,
            t_zero, period, 
            K1, fs, fc,
            V0, dV0,
            incl,
            E_tol, 
            loglike)
        
        # let's synchronise to ensure it's finished
        numba.cuda.synchronize() 

        # Now reduce the loglike array
        return sum_reduce(loglike)
    '''




'''
time = np.linspace(2450000, 2450001, 100000)
RV = rv(time)
RV = np.random.normal(RV, 1) 
RV_err = np.random.uniform(0.5,1.5, RV.shape[0])
J = 0.2 

CPU_loglike = rv(time, RV, RV_err, J=J) 
print('CPU loglike = ', CPU_loglike)

# Now copy stuff over to GPU for compliance 
d_time = numba.cuda.to_device(time)
d_RV = numba.cuda.to_device(RV)
d_RV_err = numba.cuda.to_device(RV_err)
loglike = np.empty_like(time) 
d_loglike = numba.cuda.to_device(loglike)

threads_per_block=512 
blocks = int(np.ceil(time.shape[0]/threads_per_block))
GPU_loglike = rv(d_time, d_RV, d_RV_err, J=J, gpu=1, loglike=d_loglike, blocks=blocks, threads_per_block=threads_per_block) 
print('GPU loglike = ', GPU_loglike)



# Time it 
timeit GPU_loglike = rv(d_time, d_RV, d_RV_err, J=J, gpu=1, loglike=d_loglike, blocks=blocks, threads_per_block=threads_per_block) 



plt.plot(time,RV)

plt.figure() 
K1 = np.linspace(5,15, 100)
for i in range(len(K1)) : plt.scatter(K1[i] , rv(time, RV, RV_err, J=J, K1=K1[i]))
plt.show()
'''