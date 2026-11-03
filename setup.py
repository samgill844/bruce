from setuptools import setup, Extension, find_packages
import os
import numpy
import platform

# Detect platform and compiler
system = platform.system().lower()
compiler = os.environ.get("CC", "")
is_clang = "clang" in compiler or system == "darwin"

# Keep IEEE NaN/Inf behaviour: invalid likelihoods must remain detectable.
compile_args = ["-std=c99", "-O3"]
link_args = []
include_dirs = [numpy.get_include()]
use_openmp = os.environ.get("BRUCE_OPENMP", "auto").lower()
if use_openmp not in {"auto", "0", "1"}:
    raise ValueError("BRUCE_OPENMP must be auto, 0 or 1")
if use_openmp != "0":
    if is_clang:
        prefixes = [os.environ.get("BRUCE_LIBOMP_PREFIX", ""),
                    "/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
        omp_prefix = next((p for p in prefixes if p and
                           os.path.isfile(os.path.join(p, "include", "omp.h"))), None)
        if omp_prefix:
            include_dirs.append(os.path.join(omp_prefix, "include"))
            compile_args += ["-Xpreprocessor", "-fopenmp"]
            link_args += [f"-L{omp_prefix}/lib", f"-Wl,-rpath,{omp_prefix}/lib", "-lomp"]
        elif use_openmp == "1":
            raise RuntimeError("Set BRUCE_LIBOMP_PREFIX to the libomp installation")
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

bruce_c_module = Extension(
    "bruce_c",
    sources=["src/bruce_c/bruce_numpy.c", "src/bruce_c/bruce_src.c"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="bruce",
    version="1.0.0",
    license="GPL-3.0-or-later",
    license_files=["LICENSE", "NOTICE.md"],
    description="A fast-as-hell binary star model using NumPy-C API with OpenMP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[bruce_c_module],
    python_requires=">=3.9",
    install_requires=["numpy", "scipy", "astropy"],
    extras_require={
        "analysis": ["matplotlib", "emcee", "corner", "multiprocess", "h5py",
                     "lmfit", "astroquery", "astroplan", "lightkurve", "tqdm"],
        "gui": ["pyqtgraph", "PyQt5"],
    },

    entry_points={
        "console_scripts": [
            "tesstpf = utils.tesstpf:main",
            "spocfit = utils.spocfit:main",
            "predict = utils.predict:main",
            "j2d = utils.j2d:main",
            "tj2d = utils.tj2d:main",
            "monofind = utils.monofind:main",
            "ngtsstack = utils.ngts_stack:main",
        ],
    },
)