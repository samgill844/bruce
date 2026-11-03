"""Reproducible BRUCE microbenchmark; workers isolate OpenMP thread counts.

Run from the repository root with the freshly built extension on PYTHONPATH.
Timing excludes imports, allocation of input data, I/O and optimisation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]


def measure(function, repeats, target_seconds):
    function()  # Initialise runtime and warm caches before timing.
    start = perf_counter()
    function()
    elapsed = max(perf_counter() - start, 1e-9)
    number = min(100000, max(1, int(target_seconds / elapsed)))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(number):
            function()
        samples.append((perf_counter() - start) / number)
    return dict(seconds_per_call=samples, calls_per_repeat=number)


def worker(args):
    import numpy as np
    import bruce
    import bruce_c

    rows = []
    for case, eccentricity, cadence, oversample in [
        ('circular', 0., 0., 1),
        ('eccentric', .4, 0., 1),
        ('integrated', 0., 2 / 1440, 11),
    ]:
        for size in args.sizes:
            time = np.linspace(-.15, .15, size, dtype=np.float64)
            pars = dict(t_zero=0., period=3., radius_1=.1, k=.1,
                        incl=np.arccos(.03), e=eccentricity, w=.7,
                        c=.7, alpha=.4, cadence=cadence,
                        noversample=oversample, accurate_tp=0)
            truth = bruce.binarystar.lc(time, **pars)
            rng = np.random.default_rng(42)
            error = np.full(size, .001)
            flux = truth + rng.normal(0., error)

            def model():
                return bruce.binarystar.lc(time, **pars)

            def fused():
                return bruce.binarystar.lc(time, flux, error,
                                           offset=0, **pars)

            def separate():
                residual = (flux - model()) / error
                return .5 * np.sum(residual**2)

            # The extension returns negative log likelihood, omitting constants.
            if not np.all(np.isfinite(truth)):
                raise RuntimeError('Non-finite model')
            np.testing.assert_allclose(fused(), separate(), rtol=1e-10, atol=1e-8)
            for name, function in [('model', model), ('fused_nll', fused),
                                    ('model_numpy_nll', separate)]:
                timing = measure(function, args.repeats, args.target_seconds)
                rows.append(dict(case=case, size=size, operation=name,
                                 threads=int(os.environ['OMP_NUM_THREADS']),
                                 **timing))
    return dict(rows=rows, numpy=np.__version__, extension=str(bruce_c.__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'paper/benchmarks/results.json')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 1000, 10000, 100000])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--target-seconds', type=float, default=.025)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if any(value < 1 for value in args.threads + args.sizes) or args.repeats < 3:
        parser.error('Positive sizes/thread counts and at least three repeats required')
    result = dict(metadata=dict(
        platform=platform.platform(), processor=platform.processor(),
        python=sys.version, logical_cpus=os.cpu_count(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        seed=42, repeats=args.repeats, target_seconds=args.target_seconds,
        accurate_tp=0, offset=0, fixed_uncertainty=.001, exposure_subsamples=11,
        timing='perf_counter; warm calls; median/IQR of repeated batches',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ROOT / 'setup.py', ROOT / 'src/bruce_c/bruce_src.c',
                                 ROOT / 'src/bruce_c/bruce_numpy.c', Path(__file__)]},
    ), rows=[])
    for key in ['machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu']:
        try:
            result['metadata'][key] = subprocess.check_output(
                ['sysctl', '-n', key], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    for threads in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--sizes', *map(str, args.sizes), '--repeats', str(args.repeats),
                   '--target-seconds', str(args.target_seconds)]
        print(f'Benchmarking {threads} OpenMP thread(s)', flush=True)
        data = json.loads(subprocess.check_output(command, env=env, text=True))
        result['rows'].extend(data.pop('rows'))
        result['metadata'].update(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
