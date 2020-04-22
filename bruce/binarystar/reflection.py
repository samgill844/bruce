import math, numba
from .kepler import getTrueAnomaly


@numba.njit
def xyz_planet(time, t_zero, period, sini, ecc, omdeg, xyz):
    """
    Position of the planet in Cartesian coordinates.
    The position of the ascending node is taken to be Omega=0 and the
    semi-major axis is taken to be a=1.
    :param t: time of observation (scalar or array)
    :param tzero: time of inferior conjunction, i.e., mid-transit
    :param P: orbital period
    :param sini: sine of orbital inclination
    :param ecc: eccentricity (optional, default=0)
    :param omdeg: longitude of periastron in degrees (optional, default=90)
    N.B. omdeg is the longitude of periastron for the star's orbit
    :returns: (x, y, z)
    :Example:
    
    >>> from pycheops.funcs import phase_angle
    >>> from numpy import linspace
    >>> import matplotlib.pyplot as plt
    >>> t = linspace(0,1,1000)
    >>> sini = 0.9
    >>> ecc = 0.1
    >>> omdeg = 90
    >>> x, y, z = xyz_planet(t,0,1,sini,ecc,omdeg)
    >>> plt.plot(x, y)
    >>> plt.plot(x, z)
    >>> plt.show()
        
    """
    if ecc == 0:
        nu = 2*math.pi*(time-t_zero)/period
        r = 1
        cosw = 0
        sinw = -1
    else:
        nu = getTrueAnomaly(time, ecc, omdeg, period,  t_zero, math.asin(sini), 1e-5, 0.2 )
        r = (1-ecc**2)/(1+ecc*math.cos(nu))
        omrad = math.pi*omdeg/180
        # negative here since om_planet = om_star + pi
        cosw = -math.cos(omrad)
        sinw = -math.sin(omrad)
    sinv = math.sin(nu) 
    cosv = math.cos(nu)
    cosi = math.sqrt(1-sini**2)
    xyz[0] = r*(-sinv*sinw + cosv*cosw)
    xyz[1] = r*cosi*(cosv*sinw + sinv*cosw)
    xyz[2] = -r*sini*(cosw*sinv + cosv*sinw)


@numba.njit
def reflection(t, T_0, P, A_g, r_p, ecc, om, sini, xyz):
    xyz_planet(t, T_0, P, sini, ecc, om, xyz)
    r = math.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2)
    beta = math.acos(-xyz[2]/r)
    Phi_L = (math.sin(beta) + (math.pi-beta)*math.cos(beta) )/math.pi
    return A_g*(r_p/r)**2*Phi_L 



'''
import matplotlib.pyplot as plt 
import numpy as np 


t = np.linspace(0,1,100)
xyz = np.zeros(3, dtype = np.float32)
A_g = 0.1
radius_1 = 0.2
incl = 90.*np.pi/180
sini = np.sin(incl)
for i in range(t.shape[0]):
    plt.scatter(t[i], 1 + _reflection(t[i], 0.,1., A_g, radius_1, 0., 0., sini, xyz))
plt.show()
'''