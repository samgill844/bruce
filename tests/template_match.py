import numpy as np
import matplotlib.pyplot as plt
import bruce
from scipy.signal import find_peaks

np.random.seed(0)
night = np.arange(-0.3,0.3, 0.5/24/4)

nnights = 20
t = np.array([])
for i in range(nnights) :
    t = np.concatenate((t,night+i))


fe = np.random.uniform(3e-4,4e-4,t.shape[0]).astype(np.float64)
f = np.random.normal(1,fe).astype(np.float64)
f = f + 0.003*np.sin(2*np.pi*t/5)
w = np.ones(t.shape[0]).astype(np.float64) + 0.003*np.sin(2*np.pi*t/5)
w = bruce.data.median_filter(t,f, 0.2)
w = bruce.data.convolve_1d(t,w,0.2)

period = 9
model = bruce.binarystar.lc(t, t_zero=3.1, period = period, radius_1 = 0.03, k = 0.05)
f = f*model

time_trial, DeltaL = bruce.template_match.template_match_lightcurve(t, f, fe, w, period = period,
        radius_1=0.03, k = 0.05, incl=np.pi/2,
        e=0., w = np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=5, noversample=10,
        light_3=0.,
        ld_law = -2,
        accurate_tp=1,
		jitter=0., offset=0,
		time_step=None, time_trial=None)


peaks, meta = find_peaks(DeltaL, height=50)
model1 = w*bruce.binarystar.lc(t, t_zero=time_trial[peaks[0]], period = period, radius_1 = 0.03, k = 0.05, ld_law=-2)*bruce.binarystar.lc(t, t_zero=time_trial[peaks[1]], period = period, radius_1 = 0.03, k = 0.05, ld_law=-2)

probabilities, heights = bruce.template_match.get_delta_loglike_height_from_fap(p_value=[0.01,0.001,0.0001], df=3)

print(peaks)
mosaic = '''AA
BB
CD'''
fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize = (7,7))
for i in ['A','C','D']:
    ax[i].scatter(t, f, c='k', s=1)
    ax[i].plot(t, model1, c='r', label='Template Matching', lw=1)
    ax[i].plot(t, w, 'r--', label='normalisation model', lw=1)
    ax[i].set(xlabel='Time [d]', ylabel='Flux')
    if i=='A' : ax[i].legend(fontsize=5)

ax['B'].plot(time_trial, DeltaL, c='k', lw=1)
for i in range(len(probabilities)):
    ls = ['dashed', 'dotted', 'dashdot', 'solid'][i]
    ax['B'].axhline(heights[i], c='k', ls=ls, label='FAP={:.2f}%'.format(100*probabilities[i]))
ax['B'].set_ylim(0,None)
ax['B'].legend(fontsize=5)
ax['B'].set(xlabel='Time [d]', ylabel=r'$\Delta\,\log \mathcal{L}$')

ax['C'].set_xlim((time_trial[peaks[0]] - 0.1, time_trial[peaks[0]] + 0.1))
ax['D'].set_xlim((time_trial[peaks[1]] - 0.1, time_trial[peaks[1]] + 0.1))


plt.tight_layout()
plt.savefig('../images/template_match.png')
plt.show()