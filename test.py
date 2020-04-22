import numpy as np
import matplotlib.pyplot as plt 
from bruce.binarystar import lc 


t = np.linspace(-10,10,3000)

f1 = lc(t)
f2 = lc(t, fs=0.1, fc = 0.1)

plt.plot(t,f1, c='k')
plt.plot(t,f2 - 0.01, c='r')
plt.show()
