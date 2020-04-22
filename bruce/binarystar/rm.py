import numpy as np 
import matplotlib.pyplot as plt 
import math

from bruce.binarystar.kepler import getTrueAnomaly, get_z
from bruce.binarystar.qpower2 import Flux_drop_analytical_power_2

def unsold(x, epsilon= 0.6 ):
    beta = epsilon/(1-epsilon)
    xc = np.clip(x, -1,1)
    return (2*np.sqrt(1-xc**2) / np.pi + 0.5*beta*(1-xc**2) )/ ( 1 + 2*beta/3)

def area(d,x,R):
    if (x + R < d) : return 0.
    arg1 = (d*d + x*x - R*R)/(2.*d*x)
    arg2 = (d*d + R*R - x*x)/(2.*d*R)
    arg3 = max((-d + x + R)*(d + x - R)*(d - x + R)*(d + x + R), 0.)
    if(x <= R - d)      :  return math.pi*x*x;							# planet completely overlaps stellar circle
    elif(x >= R + d)    :  return math.pi*R*R;						    # stellar circle completely overlaps planet
    else                :  return x*x*math.acos(arg1) + R*R*math.acos(arg2) - 0.5*math.sqrt(arg3) #partial overlap

time = np.linspace(-0.1,0.1,1000)

e = 0.
w = 90.*math.pi/180 
period = 1 
t_zero = 0. 
E_tol = 1e-5 
radius_1 = 0.2
b = 0.9
k = 0.05
incl = math.acos(b*radius_1)
incl_rot = 0.2

Vsini = 10

for i in range(time.shape[0]) : 
    nu = getTrueAnomaly(time[i], e, w, period,  t_zero, incl, E_tol, radius_1 )
    z = get_z(nu, e, incl, w, radius_1) 

    # Model = 
    #   Vsini * Area   *  I / I0
    A = area(z, 1,k)
    r = math.sqrt(np.clip(z**2 - b**2, 0, np.inf)) 
    Vsini_ =   unsold(r, epsilon= 0.6 ) / unsold(0, epsilon= 0.6 )
    Vsini_ = 1-Vsini_ 
    if nu < 0 : Vsini_=-Vsini_ 

    I = Flux_drop_analytical_power_2(z, k, 0.8,0.8, 1e-8)
    I0 = Flux_drop_analytical_power_2(0, k, 0.8,0.8, 1e-8)


    print(time[i], I)
    plt.scatter(time[i], math.cos(b**2)*A*Vsini*Vsini_*I/I0, c='k', s=10)
    plt.scatter(time[i], A*Vsini*Vsini_*I/I0, c='r', s=10)

plt.show()
#getTrueAnomaly(time, e, w, period,  t_zero, incl, E_tol, radius_1 )
#get_z(nu, e, incl, w, radius_1)