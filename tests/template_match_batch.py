import numpy as np
import matplotlib.pyplot as plt
import bruce
from scipy.signal import find_peaks

np.random.seed(0)
night = np.arange(-0.3,0.3, 0.5/24/4)

nnights = 50
t = np.array([])
for i in range(nnights) :
    t = np.concatenate((t,night+i))


fe = np.random.uniform(3e-4,4e-4,t.shape[0]).astype(np.float64)
f = np.random.normal(1,fe).astype(np.float64)
f = f + 0.003*np.sin(2*np.pi*t/5)
w = np.ones(t.shape[0]).astype(np.float64) + 0.003*np.sin(2*np.pi*t/5)
w = bruce.data.median_filter(t,f, 0.2)
w = bruce.data.convolve_1d(t,w,0.2)


model = bruce.binarystar.lc(t, t_zero=3.1, period = 9, radius_1 = 0.03, k = 0.05)
model = model*bruce.binarystar.lc(t, t_zero=5.9, period = 1.2, radius_1 = 0.05, k = 0.04)
model = model*bruce.binarystar.lc(t, t_zero=14.9, period = 14.3, radius_1 = 0.03, k = 0.07)
periods_ref = [9, 1.2,14.3]
f = f*model


radius_1 = np.linspace(0.01,0.2,30)
k = np.linspace(0.01,0.2,30)
b = np.linspace(0,2,30)
incl = bruce.sampler.incl_from_radius_1_b(b, 0.1)

time_trial, DeltaL = bruce.template_match.template_match_lightcurve(t, f, fe, w, period = 30,
        radius_1=radius_1, k = k, incl=incl,
        e=0., w = np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=5, noversample=10,
        light_3=0.,
        ld_law = -2,
        accurate_tp=1,
		jitter=0., offset=0,
		time_step=None, time_trial=None)


probabilities, heights = bruce.template_match.get_delta_loglike_height_from_fap(p_value=[0.01,0.001,0.0001], df=3)
peaks, meta = find_peaks(DeltaL.max(axis=(1,2,3)), height=heights[2])


periods = np.linspace(0.5,20,100000).astype(np.float64)
dispersion = bruce.template_match.phase_disperison(time_trial, peaks,  periods)



mosaic = '''
AAA
BBB
CCC
DDD'''
fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize = (7,7))
ax['A'].scatter(t, f, c='k',s=1)
ax['A'].plot(t, w, c='r', lw=1)
ax['B'].scatter(t, f/w, c='k',s=1)
ax['C'].plot(time_trial, DeltaL.max(axis=(1,2,3)), c='k',lw=1)
for i in range(len(probabilities)):
    ls = ['dashed', 'dotted', 'dashdot', 'solid'][i]
    ax['C'].axhline(heights[i], c='k', ls=ls, label='FAP={:.2f}%'.format(100*probabilities[i]))
ax['C'].set_ylim(0,None)
ax['C'].legend(fontsize=5)


ax['D'].plot(periods, dispersion, c='k',lw=1)
for p in periods_ref : ax['D'].axvline(p, c='r',lw=1, ls='--')
plt.show()




# Now lets get the parameters
parameters = []
for i in range(len(peaks)):
    idxs = np.unravel_index(DeltaL[peaks[i]].argmax(), DeltaL[peaks[i]].shape)
    parameters.append([radius_1[idxs[0]], k[idxs[1]], incl[idxs[2]]])
parameters = np.array(parameters)


print(parameters)