import numpy as np
import bruce_c

N = int(1e4)


night = np.linspace(0,0.6, N).astype(np.float64)
t = night.copy()
for i in range(1,30) : t = np.concatenate((t,night+i))
f = np.random.normal(1,4e-3,t.shape[0]).astype(np.float64)
f = f + 0.003*np.sin(2*np.pi*t/5)
bin_size = 0.5/24/3

segments = np.arange(0,N*3).astype(int)
segments = np.array_split(segments, 3)

t_bin, f_bin ,fe_bin, count = bruce_c.bin_data(t, f, bin_size)
w = bruce_c.median_filter(t,f,0.1)
w1 = bruce_c.convolve_1d(t,w,0.2)

import matplotlib.pyplot as plt
plt.scatter(t, f, c='k', s=1, alpha = 0.1)
plt.errorbar(t_bin, f_bin, yerr=fe_bin, fmt='r.')

count = 0
for seg in segments :
    plt.plot(t[seg],w[seg],'gray', zorder=10, label='median_filtered' if count == 0 else '')
    plt.plot(t[seg],w1[seg],'black', ls='--',zorder=11, label='convolved' if count == 0 else '')
    count += 1
plt.xlabel('Time [day]')
plt.ylabel('Flux')
plt.legend()
plt.tight_layout()
plt.savefig('../images/binning_test.png')
plt.show()