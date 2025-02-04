from setuptools import setup, Extension, find_packages
import numpy

bruce_c_module = Extension(
    'bruce_c',  # Module name
    sources=['src/bruce_c/bruce_numpy.c', 'src/bruce_c/bruce_src.c'],  # Sources for the extension
    include_dirs=[numpy.get_include()],  # Include directories
    extra_compile_args=['-Xpreprocessor', '-fopenmp', '-I/usr/local/include'],  # OpenMP flags
    extra_link_args=['-lomp', '-L/usr/local/lib'],  # Link OpenMP library
)

setup(
    name='bruce',
    version='1.0',
    description='A module for multiplying NumPy arrays with OpenMP',
    packages=find_packages(where='src'),  # Find Python packages in src/
    package_dir={'': 'src'},  # Set the root for Python packages
    ext_modules=[bruce_c_module],
    scripts=['utils/lcmatch']
)
