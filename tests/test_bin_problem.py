import numpy as np
import bruce, bruce_c
S27_ = np.loadtxt('/Users/sam//Documents/Papers/TIC-100504488/data_clean/TESS_SPOC_S27_flat.dat').astype(np.float64)
S27_ = S27_[np.argsort(S27_[:,0])]

t, f, fe = S27_.T




t_bin, f_bin, fe_bin, count = bruce.data.bin_data(t,f, 0.5/24)


import matplotlib.pyplot as plt
plt.scatter(t,f, c='k', s=1, alpha = 0.4)
plt.scatter(t_bin, f_bin, c='r', s=1)
plt.show()