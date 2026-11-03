"""Evaluate and fit a synthetic transit; save the figure beside the docs."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from bruce.binarystar import lc

rng = np.random.default_rng(42)
time = np.linspace(-0.15, 0.15, 401)
parameters = dict(t_zero=0.0, period=3.0, radius_1=0.1, k=0.1,
                  incl=np.arccos(0.03), e=0.0, w=np.pi/2,
                  c=0.7, alpha=0.4, cadence=2/1440, noversample=11)
truth = lc(time, **parameters)
flux_err = np.full(time.size, 0.0005)
flux = truth + rng.normal(0, flux_err)

# Fit epoch and radius ratio; hold the other quantities fixed for this tutorial.
def objective(theta):
    epoch, radius_ratio = theta
    if not (-0.03 < epoch < 0.03 and 0.01 < radius_ratio < 0.3):
        return np.inf
    trial = dict(parameters, t_zero=epoch, k=radius_ratio)
    return lc(time, flux=flux, flux_err=flux_err, offset=1, **trial)

result = minimize(objective, [0.002, 0.11], method='Nelder-Mead',
                  options={'xatol': 1e-9, 'fatol': 1e-6})
if not result.success:
    raise RuntimeError(result.message)
best = lc(time, **dict(parameters, t_zero=result.x[0], k=result.x[1]))
assert np.isfinite(best).all()
print(f'Best epoch={result.x[0]:.6f} d; k={result.x[1]:.6f}')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5),
                         gridspec_kw={'height_ratios': [3, 1]})
axes[0].errorbar(time, flux, flux_err, fmt='.', alpha=0.4, label='Synthetic data')
axes[0].plot(time, best, color='black', label='BRUCE best fit')
axes[0].set(ylabel='Relative flux', title='Direct transit fitting')
axes[0].legend()
axes[1].plot(time, 1e6*(flux-best), '.', ms=3)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set(xlabel='Time from input epoch [days]', ylabel='Residual [ppm]')
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1]/'_static/transit.png', dpi=160)
plt.close(fig)
