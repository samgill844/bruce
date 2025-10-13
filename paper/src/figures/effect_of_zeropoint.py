# Imports 
import numpy as np, bruce
np.random.seed(1)
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib
viridis = matplotlib.colormaps['viridis']
from mpl_toolkits.axes_grid1 import make_axes_locatable





# Create the lightcurve 
time = np.arange(0.9,1.1,0.5/24/6, dtype=np.float64)
width = bruce.binarystar.transit_width(0.2, 0.2, b=0, period=1)
model = bruce.binarystar.lc(time, t_zero=1, ld_law=-2,)
flux_err = np.random.uniform(1e-3, 4e-3, time.shape[0])
flux = np.random.normal(model,flux_err)
w = np.ones(time.shape[0])



# Make the plot
fig, ax = plt.subplots(1,2, figsize=(8,3))
zp = np.linspace(0.97, 1.03, 7) 
time_trial = np.linspace(0.9,1.1, 1000)
for i in range(len(zp)) :
    c = viridis((zp[i] - zp.min()) / (zp.max() - zp.min()))
    normalisation_model = zp[i]*np.ones(time.shape[0])
    time_trial, LL = bruce.template_match.template_match_lightcurve(time, flux, flux_err, normalisation_model, offset=False, time_trial=time_trial)
    ax[0].plot(time_trial, LL, color=c, ls='dashdot' if zp[i]==1 else 'solid')
    
divider = make_axes_locatable(ax[1])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=matplotlib.cm.viridis, orientation='vertical')
cb1.vmax=4
cb1.vmin=-4
cb1.set_ticks(np.linspace(0,1,zp.shape[0]))
cb1.set_ticklabels(zp)
cb1.set_label('Photometric zero-point')
#ax[0].set_yscale('symlog')
plt.gcf().add_axes(ax_cb)


ax[1].errorbar(time,flux,yerr=flux_err, fmt='.', zorder=0, color='grey', alpha=0.3)
for i in range(len(zp)) : ax[1].plot(time,zp[i]*model,color=viridis(np.linspace(0,1,zp.shape[0])[i]), lw=1, zorder=3, ls='dashdot' if zp[i]==1 else 'solid')

ax[0].set(xlabel = ('Time'), ylabel=(r'$\Delta \log \mathcal{L}$'))
ax[1].set(xlabel = ('Time'), ylabel=('Flux'))

ax[0].axhline(0, c='k',ls='--')
plt.tight_layout()
plt.subplots_adjust(left=0.09, bottom = 0.14, right = 0.93, top=0.98)
plt.savefig('effect_of_zeropoint.pdf')
plt.show()

