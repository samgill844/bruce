"""Figure-3-style comparison using the SAME power-2 law in BRUCE and ellc.

Geometry follows Maxted & Gill (2019), Fig. 3: Rstar/a=0.05, Rp/Rstar=0.1,
b=0 and e=0. The reference here is numerical integration of the power-2 law,
not the STAGGER mugrid atmosphere profile used in the published figure.

The CHEOPS coefficients are the T_eff=6000 K, log g=4.5, [Fe/H]=0 row of
Maxted (2018), CDS J/A+A/616/A39/table2 (also distributed by pycheops as
`data/limbdarkening/power2.dat`: CH 6000 4.5 +0.0 0.715452 0.643777 ...).
No network access or pycheops import is required to reproduce this comparison.
"""
import hashlib
from importlib import metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ellc
import bruce
import bruce_c

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent


def main():
    radius, ratio, c, alpha = .05, .1, .715452, .643777
    duration = bruce.binarystar.transit_width(radius, ratio, 0., period=1.)
    phase = np.linspace(-.55 * duration, .55 * duration, 1001)
    common = dict(radius_1=radius, radius_2=radius*ratio, sbratio=0.,
                  incl=90., period=1., t_zero=0., light_3=0.,
                  ld_1='power-2', ldc_1=[c, alpha],
                  shape_1='sphere', shape_2='sphere', verbose=0)
    reference = ellc.lc(phase, grid_1='very_fine', grid_2='very_fine', **common)
    fine = ellc.lc(phase, grid_1='fine', grid_2='fine', **common)
    sparse = ellc.lc(phase, grid_1='sparse', grid_2='sparse', **common)
    analytical = bruce.binarystar.lc(
        phase, radius_1=radius, k=ratio, incl=np.pi/2,
        t_zero=0., period=1., e=0., w=np.pi/2,
        c=c, alpha=alpha, cadence=0., accurate_tp=0,
    )
    for flux in [reference, fine, sparse, analytical]:
        if not np.all(np.isfinite(flux)):
            raise RuntimeError('Non-finite comparison model')
    residual = (analytical-reference)*1e6
    convergence = (fine-reference)*1e6
    sparse_residual = (sparse-reference)*1e6
    data = PAPER / 'benchmarks'
    data.mkdir(exist_ok=True)
    np.savetxt(data / 'ellc_comparison.csv',
               np.column_stack([phase, reference, analytical, sparse, fine,
                                residual, sparse_residual, convergence]),
               delimiter=',', header='phase,ellc_very_fine,bruce,ellc_sparse,ellc_fine,'
               'bruce_minus_reference_ppm,sparse_minus_reference_ppm,fine_minus_reference_ppm')
    result = dict(
        geometry=dict(radius_1=radius, k=ratio, impact_parameter=0., e=0.,
                      period_days=1., t_zero=0., exposure_days=0.),
        power2=dict(c=c, alpha=alpha, passband='CHEOPS',
                    coefficient_source='Maxted (2018), CDS J/A+A/616/A39/table2',
                    teff_K=6000, logg=4.5, metallicity=0.),
        samples=phase.size, phase_limits=[float(phase[0]), float(phase[-1])],
        reference='ellc, spherical bodies, power-2 law, very_fine grids',
        max_absolute_residual_ppm=float(np.max(np.abs(residual))),
        rms_residual_ppm=float(np.sqrt(np.mean(residual**2))),
        mean_residual_ppm=float(np.mean(residual)),
        max_fine_minus_very_fine_ppm=float(np.max(np.abs(convergence))),
        max_sparse_minus_very_fine_ppm=float(np.max(np.abs(sparse_residual))),
        numpy=np.__version__, ellc=ellc.__version__, extension=bruce_c.__file__,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), ROOT/'src/bruce_c/bruce_src.c',
                                 ROOT/'src/bruce_c/bruce_numpy.c']},
        scope='Matched-law comparison inspired by Fig. 3 of Maxted & Gill (2019); '
              'not a reproduction of the published STAGGER-profile reference. '
              'No parameters are fitted or adjusted to reduce residuals. '
              'The K2 expression in BRUCE is unchanged.',
    )
    (data/'ellc_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.5), sharex=True,
                            gridspec_kw={'height_ratios':[2,1], 'hspace':.08})
    axes[0].plot(phase, reference, color='.2', lw=1.6,
                 label='ellc numerical power-2 (very fine)')
    axes[0].plot(phase, analytical, '--', color='#185b87', lw=1.1,
                 label='BRUCE analytical power-2')
    axes[0].set_ylabel('Relative flux')
    axes[0].legend(frameon=False, fontsize=8, loc='upper center')
    axes[1].plot(phase, sparse_residual, '--', color='#73904c', lw=1.,
                 label='ellc sparse - reference')
    axes[1].plot(phase, residual, color='#185b87', lw=1.1,
                 label='BRUCE - reference')
    axes[1].axhline(0, color='.5', lw=.6)
    axes[1].set(xlabel='Orbital phase', ylabel='Difference (ppm)')
    axes[1].legend(frameon=False, fontsize=7.5, loc='lower center', ncol=2)
    for ext in ['png','pdf']:
        fig.savefig(PAPER/'figures'/f'ellc_comparison.{ext}', bbox_inches='tight',dpi=220)
    plt.close(fig)
    print(json.dumps({key: value for key,value in result.items()
                      if 'ppm' in key}, indent=2))


if __name__ == '__main__':
    main()
