"""Single- and double-lined Keplerian RV curves (km/s in this example)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bruce.binarystar import rv1, rv2

time = np.linspace(0, 10, 501)
parameters = dict(t_zero=0.0, period=5.0, e=0.35, w=0.8,
                  incl=np.pi/2, V0=12.0, accurate_tp=1)
single = rv1(time, K1=0.15, **parameters)
double = rv2(time, K1=20.0, K2=35.0, **parameters)
print('rv2 array shapes:', [value.shape for value in double])
assert all(np.isfinite(value).all() for value in double)
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
axes[0].plot(time, single)
axes[0].set(ylabel='Primary RV [km/s]', title='Single-lined system: rv1')
axes[1].plot(time, double[0], label='Primary')
axes[1].plot(time, double[1], label='Secondary')
axes[1].set(xlabel='Time [days]', ylabel='RV [km/s]', title='Double-lined system: rv2')
axes[1].legend()
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1]/'_static/radial_velocity.png', dpi=160)
plt.close(fig)
