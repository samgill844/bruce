import bruce 
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-0.75,0.75,1000)
f = bruce.binarystar.lc(t)
noise = 1e-2
fe = np.random.uniform(0.5*noise,1.5*noise, t.shape[0])
f = np.random.normal(f,fe)
w = np.ones(t.shape)
time_trial, z, S, rho2 = bruce.template_match.template_match_snr(t, f, fe, w)

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(time_trial, z, c='k',lw=1)
ax[1].errorbar(t,f, yerr=fe, fmt='k.', markersize=1)
ax[1].plot(time_trial, bruce.binarystar.lc(time_trial), c='r', lw=1)
ax[0].set(ylabel='z')
ax[1].set(ylabel='Flux', xlabel='Time')

plt.show()