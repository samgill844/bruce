import numpy as np
import bruce_c

N = int(1e4)


night = np.linspace(0,0.6, N).astype(np.float64)
t = night.copy()
for i in range(1,10) : t = np.concatenate((t,night+i))
f = np.random.normal(1,4e-3,t.shape[0]).astype(np.float64)
f = f + 0.003*np.sin(2*np.pi*t/5)
bin_size = 0.5/24/3

t_bin, f_bin ,fe_bin, count = bruce_c.bin_data(t, f, bin_size)
w = bruce_c.median_filter(t,f,0.1)
w1 = bruce_c.convolve_1d(t,w,0.3)

import matplotlib.pyplot as plt
plt.scatter(t, f, c='k', s=1, alpha = 0.1)
plt.errorbar(t_bin, f_bin, yerr=fe_bin, fmt='r.')
plt.plot(t,w,'gray')
plt.plot(t,w1,'gray', ls='--')

plt.show()