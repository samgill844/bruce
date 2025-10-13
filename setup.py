from setuptools import setup, Extension, find_packages
import sys
import os
import numpy
import platform

# Detect platform and compiler
system = platform.system().lower()
compiler = os.environ.get("CC", "")
is_clang = "clang" in compiler or system == "darwin"

# Base compile args
compile_args = ["-O3", "-ffast-math"]
link_args = []

# Try to enable OpenMP safely
if is_clang:
    # macOS Clang (Xcode) often lacks OpenMP, but Homebrew LLVM supports it
    compile_args += ["-Xpreprocessor", "-fopenmp"]
    link_args += ["-lomp"]
    omp_include = "/usr/local/include"
    omp_lib = "/usr/local/lib"
    if os.path.exists(omp_include):
        compile_args += [f"-I{omp_include}"]
    if os.path.exists(omp_lib):
        link_args += [f"-L{omp_lib}"]
else:
    # GCC and compatible compilers
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]

# Add architecture optimizations if supported
if not is_clang:
    compile_args += ["-march=native"]
else:
    compile_args += ["-mavx2"]

bruce_c_module = Extension(
    "bruce_c",
    sources=["src/bruce_c/bruce_numpy.c", "src/bruce_c/bruce_src.c"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="bruce",
    version="1.0.0",
    description="A fast-as-hell binary star model using NumPy-C API with OpenM.P",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[bruce_c_module],
    scripts=["utils/lcmatch", "utils/tesstpf", "utils/spocfit"],
)
