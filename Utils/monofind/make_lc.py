import numpy as np 
import matplotlib.pyplot as plt 
from bruce.binarystar import lc 

t = np.arange(0,30, 0.5/24)
f = lc(t, period = 30, radius_1 = 0.01, t_zero = 17, SBR = 0.7)
m = -2.5*np.log10(f)
me = np.random.uniform(1e-3, 3e-3, m.shape[0])
m = np.random.normal(m,me) + np.sin(2*np.pi*t / 12)*4e-3

tmp = np.array([t, m, me]).T
np.savetxt('monolc.dat', tmp)


plt.scatter(t, m, c='k', s=10)
plt.show()
