import numba, math

@numba.njit
def Fdoppler(alpha, alpha_doppler, K1 ):
    Ad = alpha_doppler*K1/299792.458
    return Ad*math.sin(alpha -math.pi/2.) 

'''
import numpy as np 
import matplotlib.pyplot as plt 
alpha = np.linspace(0,2*math.pi,100) 
alpha_doppler = 0.1
K1 = 150 

for i in alpha : plt.scatter(i, Fdoppler(i, alpha_doppler, K1 ) ) 
plt.show()
'''