"""Recover the epoch of one injected transit without an archive download."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bruce.binarystar import lc
from bruce.template_match import template_match_lightcurve

rng = np.random.default_rng(7)
time = np.linspace(0, 2, 1441)
epoch = 1.05
shape = dict(period=10.0, radius_1=0.035, k=0.1,
             incl=np.arccos(0.035*0.3), e=0.0, w=np.pi/2,
             c=0.7, alpha=0.4, cadence=2/1440, noversample=11,
             accurate_tp=1, ld_law=-2)
baseline = 1 + 0.001*(time-1)
flux_err = np.full(time.size, 0.001)
flux = baseline*lc(time, t_zero=epoch, **shape) + rng.normal(0, flux_err)
trial, score = template_match_lightcurve(
    time, flux, flux_err, baseline, time_trial=np.linspace(0.2, 1.8, 601),
    **shape)
best_epoch = trial[np.argmax(score)]
assert abs(best_epoch-epoch) < 0.01
print(f'Injected epoch={epoch:.4f}; recovered epoch={best_epoch:.4f} d')
fig, axes = plt.subplots(2, 1, figsize=(7, 5))
axes[0].plot(time, flux/baseline, '.', ms=2, alpha=0.5)
axes[0].axvline(epoch, ls='--', color='black', label='Injected epoch')
axes[0].set(xlabel='Time [days]', ylabel='Normalised flux')
axes[0].legend()
axes[1].plot(trial, score)
axes[1].axvline(best_epoch, color='tab:orange', ls='--')
axes[1].set(xlabel='Trial mid-transit epoch [days]', ylabel='2 delta log likelihood')
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1]/'_static/template_search.png', dpi=160)
plt.close(fig)
