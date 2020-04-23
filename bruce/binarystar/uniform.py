import math, numba 

@numba.njit
def area(d, x, R):
    if (x <= (R - d)) :  return math.pi*x*x						
    elif (x >= (R + d)) : return math.pi*R*R	
    elif (d > (x+R)) : return 0.			
    else :  
        arg1 = (d*d + x*x - R*R)/(2.*d*x)
        arg2 = (d*d + R*R - x*x)/(2.*d*R)
        arg3 = max((-d + x + R)*(d + x - R)*(d - x + R)*(d + x + R), 0.)
        return x*x*math.acos(arg1) + R*R*math.acos(arg2) - 0.5*math.sqrt(arg3)

@numba.njit
def frac_secondary(d,x,R):
    return 1 - area(d, x,R)/(math.pi*x**2)


@numba.njit
def Flux_drop_uniform( z, k, SBR):
    if SBR > 0 : 
        return -SBR*area(z, k, 1)
    else:
        if (z >= 1. + k) or ((z >= 1.) and  (z <= k - 1.)) :  return 0.0           # total eclipse of the star
        elif  (z <= 1. - k)                                :  return - SBR*k*k   # planet is fully in transit		
        else :                                                                     #  planet is crossing the limb
            kap1 = math.acos(min((1. - k*k + z*z)/2./z, 1.))
            kap0 = math.acos(min((k*k + z*z - 1.)/2./k/z, 1.))
            return - SBR*  (k*k*kap0 + kap1 - 0.5*math.sqrt(max(4.*z*z - math.pow(1. + z*z - k*k, 2.), 0.)))/math.pi 


'''
import numpy as np 
import matplotlib.pyplot as plt 
z = np.linspace(-2,2,1000) 
_z = np.abs(z)
k = 0.1
SBR = 0.4 


for i in range(len(z)) : plt.scatter(z[i], Flux_drop_uniform( _z[i], k, SBR)) 
plt.show()
'''