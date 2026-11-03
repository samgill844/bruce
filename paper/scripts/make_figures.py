"""Generate the paper figures from seeded simulations and raw benchmark data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize

import bruce

PAPER = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 220})


def save(figure, name):
    for extension in ['png', 'pdf']:
        figure.savefig(PAPER / 'figures' / f'{name}.{extension}', bbox_inches='tight')
    plt.close(figure)


def transit():
    rng = np.random.default_rng(42)
    time = np.arange(-.15, .15, 2 / 1440, dtype=np.float64)
    error = np.full(time.size, .0005)
    fixed = dict(period=3., incl=np.arccos(.03), e=0., w=np.pi / 2,
                 c=.7, alpha=.4, cadence=2 / 1440, noversample=11,
                 accurate_tp=0)

    def model(theta):
        return bruce.binarystar.lc(time, t_zero=theta[0], k=theta[1],
                                  radius_1=theta[2], **fixed)

    injected = np.array([0., .1, .1])
    truth = model(injected)
    flux = truth + rng.normal(0., error)
    result = least_squares(lambda theta: (flux - model(theta)) / error,
                           [.002, .09, .11],
                           bounds=([-.02, .02, .03], [.02, .2, .2]))
    if not result.success:
        raise RuntimeError(result.message)
    fit = model(result.x)
    np.savetxt(PAPER / 'benchmarks/synthetic_transit.csv',
               np.column_stack([time, flux, error, truth, fit]), delimiter=',',
               header='time_days,flux,flux_error,injected_model,fitted_model')
    (PAPER / 'benchmarks/synthetic_fit.json').write_text(json.dumps(dict(
        seed=42, names=['t_zero', 'k', 'radius_1'], injected=injected.tolist(),
        fitted=result.x.tolist(), fixed=fixed, success=bool(result.success),
        evaluations=result.nfev, rms_ppm=float(np.std(flux - fit) * 1e6),
        note='Illustrative fit with generating model; not an independent accuracy test.',
    ), indent=2) + '\n')
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.1), sharex=True,
                             gridspec_kw={'height_ratios': [2.3, 1], 'hspace': .08})
    hours = time * 24
    axes[0].errorbar(hours, flux, error, fmt='.', color='.65',
                     ms=3, elinewidth=.5, label='Synthetic observations')
    axes[0].plot(hours, fit, color='#185b87', lw=1.6, label='BRUCE fit')
    axes[0].set_ylabel('Relative flux')
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    axes[1].plot(hours, (flux - fit) * 1e6, '.', color='.4', ms=3)
    axes[1].axhline(0, color='#185b87', lw=1)
    axes[1].set(xlabel='Time from injected mid-transit (hours)', ylabel='Residual (ppm)')
    save(fig, 'synthetic_transit')


def performance():
    data = json.loads((PAPER / 'benchmarks/results.json').read_text())
    rows = data['rows']
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), constrained_layout=True)
    for case, label, colour in [('circular', r'$e=0$, instantaneous', '#185b87'),
                                ('eccentric', r'$e=0.4$, instantaneous', '#b66a18'),
                                ('integrated', r'$e=0$, 11 subsamples', '#6c5495')]:
        subset = sorted([r for r in rows if r['case'] == case and
                         r['operation'] == 'fused_nll' and r['threads'] == 1],
                        key=lambda r: r['size'])
        size = np.array([r['size'] for r in subset])
        values = np.array([r['seconds_per_call'] for r in subset])
        median = np.median(values, axis=1)
        axes[0].loglog(size, 1 / median, 'o-', label=label, color=colour, ms=4)
        axes[0].fill_between(size, 1 / np.percentile(values, 75, axis=1),
                             1 / np.percentile(values, 25, axis=1),
                             color=colour, alpha=.15)
        parallel = sorted([r for r in rows if r['case'] == case and
                           r['operation'] == 'fused_nll' and r['threads'] == 4],
                          key=lambda r: r['size'])
        speedup = median / np.array([np.median(r['seconds_per_call']) for r in parallel])
        axes[1].semilogx(size, speedup, 'o-', label=label, color=colour, ms=4)
    axes[0].set(xlabel='Samples per model', ylabel='Likelihood evaluations per second',
                title='Direct C likelihood, one thread')
    axes[1].axhline(1, color='.6', lw=.8, ls=':')
    axes[1].set(xlabel='Samples per model', ylabel='One-thread / four-thread time',
                title='Measured parallel speed-up')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=.15)
    axes[1].grid(alpha=.15)
    save(fig, 'performance')


def use_cases():
    """Illustrate fitting a transit with the direct C likelihood."""
    fit_data = np.loadtxt(PAPER / 'benchmarks/synthetic_transit.csv', delimiter=',')
    # Use the public fused C objective for the fit shown in panel (a).
    fit_fixed = dict(period=3., incl=np.arccos(.03), e=0., w=np.pi / 2,
                     c=.7, alpha=.4, cadence=2 / 1440, noversample=11,
                     accurate_tp=0)

    def negative_loglike(theta):
        return bruce.binarystar.lc(
            fit_data[:, 0], fit_data[:, 1], fit_data[:, 2],
            t_zero=theta[0], k=theta[1], radius_1=theta[2], offset=0,
            **fit_fixed,
        )

    c_fit = minimize(negative_loglike, [.002, .09, .11], method='Nelder-Mead',
                     bounds=[(-.02, .02), (.02, .2), (.03, .2)],
                     options={'xatol': 1e-10, 'fatol': 1e-7, 'maxiter': 2000})
    if not c_fit.success or not np.isfinite(c_fit.fun):
        raise RuntimeError('Direct C likelihood fit failed')
    fit_data[:, 4] = bruce.binarystar.lc(
        fit_data[:, 0], t_zero=c_fit.x[0], k=c_fit.x[1],
        radius_1=c_fit.x[2], **fit_fixed,
    )
    reference_nll = .5 * np.sum(((fit_data[:, 1] - fit_data[:, 4]) /
                                 fit_data[:, 2])**2)
    np.testing.assert_allclose(c_fit.fun, reference_nll, rtol=1e-10)
    np.savetxt(PAPER / 'benchmarks/synthetic_c_likelihood_fit.csv', fit_data,
               delimiter=',', header='time_days,flux,error,injected_model,c_fitted_model')
    (PAPER / 'benchmarks/synthetic_c_likelihood_fit.json').write_text(json.dumps(dict(
        method='Nelder-Mead', objective='bruce.binarystar.lc with flux and flux_err',
        offset=0, fitted=c_fit.x.tolist(), negative_log_likelihood=float(c_fit.fun),
        success=bool(c_fit.success), evaluations=int(c_fit.nfev),
        note='Synthetic same-model demonstration; fixed uncertainty and zero jitter.',
    ), indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(6.5, 2.6), constrained_layout=True)
    ax.errorbar(fit_data[:, 0] * 24, fit_data[:, 1], fit_data[:, 2],
                fmt='.', color='.65', ms=3, elinewidth=.5,
                label='Synthetic observations')
    ax.plot(fit_data[:, 0] * 24, fit_data[:, 4], color='#185b87', lw=1.5,
            label='Direct C likelihood fit')
    ax.set(xlabel='Time from injected mid-transit (hours)', ylabel='Relative flux')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    save(fig, 'use_cases')


if __name__ == '__main__':
    (PAPER / 'figures').mkdir(exist_ok=True)
    transit()
    use_cases()
    from compare_ellc import main as compare_ellc
    compare_ellc()
    performance()
