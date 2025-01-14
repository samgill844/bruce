# BRUCE
------------------
<table border="0">
 <tr>
    <td>Lightning-fast binary star model and data reduction package. This package uses the Numpy C API and OpenMP to create an incredibly fast and optimised suite of tools for the reduction and analysis of exoplanets and stellar binaries.</td>
    <td> <img src="images/bruce.png" width="400" alt="Bruce the dog"/></a> </td>
 </tr>
</table>

## Binary star model
---------
The binary star model described here is heavily influenced by [ellc](https://github.com/pmaxted) [(Maxted 2016)](https://www.aanda.org/articles/aa/full_html/2016/07/aa28579-16/aa28579-16.html) and credit should be given to this work for the functions used in this package related to solving for the mean, eccentric, and true anomaly, along with functions for the scaled orbital separation and radial velocities (this list is not exhaustive). The transit model used here is called qpower2 [Maxted and Gill 2016](https://ui.adsabs.harvard.edu/abs/2019A%26A...622A..33M/abstract) and uses the power-2 limb-darkening law. People are encouraged not to fit the limb-darkening coefficients directly but rather to use the decorelated parameters h1 and h2 described in [Maxted 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..39M/abstract).

Most functions in this package depend on a numpy array as an input and are written to expect double precision data (np.float64). Care has been taken to recast data where appropriate in the C code to avoid errors where the data is not of type np.float64, but this comes at a cost to performance. For best results, use np.float64 throughout (data, input models, etc.). The C source code uses OpenMP to multiprocess calculations where appropriate. To control this, set the OMP_NUM_THREADS environmental variable to the number of processors you wish to use. If fitting data using multiprocessing, you may encounter forking issues, and so I would suggest setting this to 1 prior to fitting.

```bash
export OMP_NUM_THREADS=12
```

