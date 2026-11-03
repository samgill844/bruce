import bruce, numpy as np
import matplotlib.pyplot as plt

night = np.arange(0.2,0.6, 13/86400)
time = np.concatenate([night+i for i in [1,3,4,5,6,8,9,13,14,15]])
flux = np.ones(time.shape[0])
flux_err = 0.1*np.ones(time.shape[0])

t = bruce.ambiguous_period.photometry_time_series(
    time=time,
    flux=flux,
    flux_err=flux_err,
    w=None,
    sky_bkg=None,
    filename=None,
)
width = 6/24
p_min = 30
p_max = 100
periods = np.linspace(p_min, p_max, 10000)
phase_width = width / periods
coverage = np.zeros(periods.shape)

for i,period in enumerate(periods):
    phase = bruce.data.phase_times(t.time,0,period, phase_offset=0.5)
    #plt.scatter(np.ones(phase.shape)*period, phase, c='k',s=1, alpha = 0.4)
    coverage[i] = phase[(phase>(-width/period/2)) & (phase<(width/period/2))].shape[0]
#plt.plot(periods, phase_width/2, 'r', lw=1, zorder=10000)
#plt.plot(periods, -phase_width/2, 'r', lw=1, zorder=10000)

#plt.show()

plt.plot(periods, coverage)
plt.show()