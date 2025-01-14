#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>

#include "bruce_src.h"

#define N_BIN_MAX 10000
// loglike function implementation with OpenMP
static PyObject* loglike(PyObject* self, PyObject* args) {
    PyArrayObject *y = NULL, *yerr = NULL, *model = NULL;
    PyArrayObject *y_cast = NULL, *yerr_cast = NULL, *model_cast = NULL;
    double jitter, offset;

    // Parse Python arguments (y, yerr, model, jitter, offset)
    if (!PyArg_ParseTuple(args, "O!O!O!dd", 
                          &PyArray_Type, &y, 
                          &PyArray_Type, &yerr, 
                          &PyArray_Type, &model, 
                          &jitter, &offset)) {
        return NULL;
    }

    // Ensure the arrays are cast to double if necessary
    if (PyArray_TYPE(y) != NPY_DOUBLE) {
        y_cast = (PyArrayObject*) PyArray_FROMANY((PyObject*) y, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
        if (!y_cast) goto fail;
        y = y_cast; // Use the casted array
    }
    if (PyArray_TYPE(yerr) != NPY_DOUBLE) {
        yerr_cast = (PyArrayObject*) PyArray_FROMANY((PyObject*) yerr, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
        if (!yerr_cast) goto fail;
        yerr = yerr_cast;
    }
    if (PyArray_TYPE(model) != NPY_DOUBLE) {
        model_cast = (PyArrayObject*) PyArray_FROMANY((PyObject*) model, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
        if (!model_cast) goto fail;
        model = model_cast;
    }

    // Check that the arrays are contiguous
    if (!PyArray_ISCONTIGUOUS(y) || !PyArray_ISCONTIGUOUS(yerr) || !PyArray_ISCONTIGUOUS(model)) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be contiguous.");
        goto fail;
    }

    // Check that the arrays have the same size
    npy_intp size = PyArray_SIZE(y);
    if (size != PyArray_SIZE(yerr) || size != PyArray_SIZE(model)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        goto fail;
    }

    // Access the data of the input arrays
    double *y_data = (double*) PyArray_DATA(y);
    double *yerr_data = (double*) PyArray_DATA(yerr);
    double *model_data = (double*) PyArray_DATA(model);

    // Initialize log-likelihood
    double loglike_val = 0.0;

    // OpenMP parallel reduction
    #pragma omp parallel for reduction(-:loglike_val)
    for (npy_intp i = 0; i < size; i++) {
        loglike_val -= bruce_loglike(y_data[i], yerr_data[i], model_data[i], jitter, offset);
    }

    // Free casted arrays (if allocated) and return result
    Py_XDECREF(y_cast);
    Py_XDECREF(yerr_cast);
    Py_XDECREF(model_cast);
    return PyFloat_FromDouble(loglike_val);

fail:
    Py_XDECREF(y_cast);
    Py_XDECREF(yerr_cast);
    Py_XDECREF(model_cast);
    return NULL;
}








// Python wrapper for the _lc function
static PyObject* lc(PyObject* self, PyObject* args) {
    PyArrayObject *time_array = NULL, *output_array = NULL;
    double t_zero, period, radius_1, k, incl, e, w, c, alpha, cadence, light_3;
    int noversample, ld_law, accurate_tp;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!ddddddddddidii", //"O!ddddddddddiidi",
                          &PyArray_Type, &time_array,  // NumPy array for time
                          &t_zero, &period, &radius_1, &k,
                          &incl, &e, &w, &c, &alpha, &cadence,
                          &noversample, &light_3, &ld_law, &accurate_tp)) {
        return NULL;  // Parsing failed
    }

    // Ensure the time array is of type double and contiguous
    PyArrayObject *time_cast = NULL;
    if (PyArray_TYPE(time_array) != NPY_DOUBLE) {
        time_cast = (PyArrayObject*) PyArray_FROMANY(
            (PyObject*) time_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (!time_cast) {
            PyErr_SetString(PyExc_TypeError, "Failed to cast time array to numpy.float64.");
            return NULL;
        }
    } else {
        time_cast = time_array;
        Py_INCREF(time_cast);  // Increment reference since we won't own it otherwise
    }

    // Get the size of the time array
    npy_intp size = PyArray_SIZE(time_cast);

    // Create an output array of the same size as the time array
    output_array = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(time_cast), NPY_DOUBLE);
    if (!output_array) {
        Py_DECREF(time_cast);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array.");
        return NULL;
    }

    // Access the data pointers for the arrays
    double *time_data = (double*) PyArray_DATA(time_cast);
    double *output_data = (double*) PyArray_DATA(output_array);

    // Perform computation for each element in the time array
    #pragma omp parallel for
    for (npy_intp i = 0; i < size; i++) {
        output_data[i] = _lc(time_data[i], t_zero, period, radius_1, k,
                             incl, e, w, c, alpha, cadence,
                             noversample, light_3, ld_law, accurate_tp);
    }

    // Clean up and return the result
    Py_DECREF(time_cast);
    return (PyObject*) output_array;
}













static PyObject* lc_loglike(PyObject* self, PyObject* args) {
    PyArrayObject *time_array = NULL, *flux_array = NULL, *flux_err_array = NULL;
    double t_zero, period, radius_1, k, incl, e, w, c, alpha, cadence, light_3, jitter, offset;
    int noversample, ld_law, accurate_tp;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!O!O!ddddddddddidiidi",
                          &PyArray_Type, &time_array,    // NumPy array for time
                          &PyArray_Type, &flux_array,    // NumPy array for flux
                          &PyArray_Type, &flux_err_array, // NumPy array for flux error
                          &t_zero, &period, &radius_1, &k,
                          &incl, &e, &w, &c, &alpha, &cadence,
                          &noversample, &light_3, &ld_law, &accurate_tp,
                          &jitter, &offset)) {
        return NULL;  // Parsing failed
    }

    // Ensure the input arrays are of type double and contiguous
    PyArrayObject *time_cast = NULL, *flux_cast = NULL, *flux_err_cast = NULL;
    if (PyArray_TYPE(time_array) != NPY_DOUBLE) {
        time_cast = (PyArrayObject*) PyArray_FROMANY(
            (PyObject*) time_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (!time_cast) {
            PyErr_SetString(PyExc_TypeError, "Failed to cast time array to numpy.float64.");
            return NULL;
        }
    } else {
        time_cast = time_array;
        Py_INCREF(time_cast);
    }

    if (PyArray_TYPE(flux_array) != NPY_DOUBLE) {
        flux_cast = (PyArrayObject*) PyArray_FROMANY(
            (PyObject*) flux_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (!flux_cast) {
            Py_DECREF(time_cast);
            PyErr_SetString(PyExc_TypeError, "Failed to cast flux array to numpy.float64.");
            return NULL;
        }
    } else {
        flux_cast = flux_array;
        Py_INCREF(flux_cast);
    }

    if (PyArray_TYPE(flux_err_array) != NPY_DOUBLE) {
        flux_err_cast = (PyArrayObject*) PyArray_FROMANY(
            (PyObject*) flux_err_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (!flux_err_cast) {
            Py_DECREF(time_cast);
            Py_DECREF(flux_cast);
            PyErr_SetString(PyExc_TypeError, "Failed to cast flux error array to numpy.float64.");
            return NULL;
        }
    } else {
        flux_err_cast = flux_err_array;
        Py_INCREF(flux_err_cast);
    }

    // Ensure arrays have the same size
    npy_intp size = PyArray_SIZE(time_cast);
    if (size != PyArray_SIZE(flux_cast) || size != PyArray_SIZE(flux_err_cast)) {
        Py_DECREF(time_cast);
        Py_DECREF(flux_cast);
        Py_DECREF(flux_err_cast);
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        return NULL;
    }

    // Access the data pointers for the arrays
    double *time_data = (double*) PyArray_DATA(time_cast);
    double *flux_data = (double*) PyArray_DATA(flux_cast);
    double *flux_err_data = (double*) PyArray_DATA(flux_err_cast);

    // Initialize log-likelihood
    double loglike_val = 0.0;

    // Perform computation in parallel using OpenMP
    #pragma omp parallel for reduction(-:loglike_val)
    for (npy_intp i = 0; i < size; i++) {
        // Compute the light curve model for this index
        double model = _lc(time_data[i], t_zero, period, radius_1, k,
                           incl, e, w, c, alpha, cadence,
                           noversample, light_3, ld_law, accurate_tp);
        loglike_val -= bruce_loglike(flux_data[i], flux_err_data[i], model, jitter, offset);
    }

    // Clean up and return the result
    Py_DECREF(time_cast);
    Py_DECREF(flux_cast);
    Py_DECREF(flux_err_cast);
    return PyFloat_FromDouble(loglike_val);
}
















static PyObject *bin_data(PyObject *self, PyObject *args) {
    PyArrayObject *time_array, *flux_array;
    double bin_size;
    if (!PyArg_ParseTuple(args, "O!O!d",
                          &PyArray_Type, &time_array,
                          &PyArray_Type, &flux_array,
                          &bin_size)) {
        return NULL;
    }

    // if (PyArray_TYPE(time_array) != NPY_DOUBLE || PyArray_TYPE(flux_array) != NPY_DOUBLE) {
    //     PyErr_SetString(PyExc_TypeError, "Input arrays must be of type numpy.float64.");
    //     return NULL;
    // }
    // Ensure input arrays are cast to NPY_DOUBLE
    PyArrayObject *time_array_cast = (PyArrayObject *)PyArray_FromAny(
        (PyObject *)time_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if (time_array_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'time_array' to numpy.float64.");
        return NULL;
    }

    PyArrayObject *flux_array_cast = (PyArrayObject *)PyArray_FromAny(
        (PyObject *)flux_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if (flux_array_cast == NULL) {
        Py_DECREF(time_array_cast);
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'flux_array' to numpy.float64.");
        return NULL;
    }

    int time_size = (int)PyArray_SIZE(time_array_cast);
    double *time = (double *)PyArray_DATA(time_array_cast);
    double *flux = (double *)PyArray_DATA(flux_array_cast);

    // Create edges array
    int binned_size =  (int) ceil((time[time_size - 1] - time[0]) / bin_size) + 1;
    double *edges = (double *)malloc((binned_size + 1) * sizeof(double));
    linspace(time[0], time[time_size - 1], binned_size + 1, edges);

     // Allocate intermediate arrays
    double *binned_flux = (double *)malloc(binned_size * sizeof(double));
    double *binned_flux_err = (double *)malloc(binned_size * sizeof(double));
    int *count = (int *)malloc(binned_size * sizeof(int));
    double *mid_times = (double *)malloc(binned_size * sizeof(double));

    bin_data_fast(time, flux, time_size, edges, binned_flux, binned_flux_err, binned_size, count);

    // Count non-zero bins
    int non_zero_bins = 0;
    for (int i = 0; i < binned_size; i++) {
        if (count[i] > 0) non_zero_bins++;
    }

    // Allocate output arrays
    npy_intp dims[1] = {non_zero_bins};
    PyArrayObject *binned_flux_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject *binned_flux_err_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject *count_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT, 0);
    PyArrayObject *mid_times_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    double *filtered_flux = (double *)PyArray_DATA(binned_flux_array);
    double *filtered_flux_err = (double *)PyArray_DATA(binned_flux_err_array);
    int *filtered_count = (int *)PyArray_DATA(count_array);
    double *filtered_mid_times = (double *)PyArray_DATA(mid_times_array);

    // Populate filtered arrays
    int idx = 0;
    int warning = 0;
    for (int i = 0; i < binned_size; i++) 
    {
        if (count[i] > 0) {
            filtered_flux[idx] = binned_flux[i];
            filtered_flux_err[idx] = binned_flux_err[i];
            filtered_count[idx] = count[i];
            filtered_mid_times[idx] = (edges[i] + edges[i+1])/2; //mid_times[i];
            idx++;
        }
        if (count[i] == N_BIN_MAX)   warning=1;
    }
    if (warning == 1){
        printf("Warning: %d data points in a bin, data not binned properly!\n", N_BIN_MAX);
    }

    // Free temporary arrays
    free(edges);
    free(binned_flux);
    free(binned_flux_err);
    free(count);
    free(mid_times);

    // Return result as a tuple
    PyObject *result = Py_BuildValue("NNNN",
                                     PyArray_Return(mid_times_array),
                                     PyArray_Return(binned_flux_array),
                                     PyArray_Return(binned_flux_err_array),
                                     PyArray_Return(count_array));
    return result;
}













static PyObject *median_filter(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    double window;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOd", &x_obj, &y_obj, &window)) {
        return NULL;
    }

    // Ensure NumPy is initialized
    import_array();

    // Cast input arrays to Numpy arrays
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROMANY(
        x_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED);
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'x' to numpy.float64 array.");
        return NULL;
    }

    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROMANY(
        y_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED);
    if (y_array == NULL) {
        Py_DECREF(x_array);
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'y' to numpy.float64 array.");
        return NULL;
    }

    // Ensure x and y have the same size
    int signal_length = (int)PyArray_SIZE(x_array);
    if (PyArray_SIZE(y_array) != signal_length) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "Input arrays 'x' and 'y' must have the same length.");
        return NULL;
    }

    // Allocate output array
    npy_intp dims[1] = {signal_length};
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (output_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        return NULL;
    }

    // Get pointers to data
    const double *x = (const double *)PyArray_DATA(x_array);
    const double *y = (const double *)PyArray_DATA(y_array);
    double *output_signal = (double *)PyArray_DATA(output_array);

    // Call the median filter
    median_filter_fast(x, y, output_signal, window, signal_length);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    // Return the output array
    return (PyObject *)output_array;
}


static PyObject *convolve_1d(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    double window;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOd", &x_obj, &y_obj, &window)) {
        return NULL;
    }

    // Ensure NumPy is initialized
    import_array();

    // Cast input arrays to Numpy arrays
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROMANY(
        x_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED);
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'x' to numpy.float64 array.");
        return NULL;
    }

    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROMANY(
        y_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED);
    if (y_array == NULL) {
        Py_DECREF(x_array);
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'y' to numpy.float64 array.");
        return NULL;
    }

    // Ensure x and y have the same size
    int signal_length = (int)PyArray_SIZE(x_array);
    if (PyArray_SIZE(y_array) != signal_length) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "Input arrays 'x' and 'y' must have the same length.");
        return NULL;
    }

    // Allocate output array
    npy_intp dims[1] = {signal_length};
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (output_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        return NULL;
    }

    // Get pointers to data
    const double *x = (const double *)PyArray_DATA(x_array);
    const double *y = (const double *)PyArray_DATA(y_array);
    double *output_signal = (double *)PyArray_DATA(output_array);

    // Call the median filter
    convolve_1d_fast(x, y, output_signal, window, signal_length);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    // Return the output array
    return (PyObject *)output_array;
}









// Define the methods in the module
static PyMethodDef bruce_c_methods[] = {
    {"loglike", loglike, METH_VARARGS, 
    "Calculate the log-likliehood of the data given the model, errors, jitter, and offset.\n"
    "Assumes y,yerr, and model are arrays of same shape. \n"
    "\n"
    "Parameters:\n"
    "    y: array of data points                              (np.float64)\n"
    "    yerr: array of errors                                (np.float64)\n"
    "    model: array of model values                         (np.float64)\n"
    "    jitter: Jitter Value added in quadrature to yerr.    (float) \n"
    "    offset: Subtract an arbritrary offset when set to 1. (int)\n"
    "\n"
    "Returns:\n"
    "    log-likelihood                                       (np.float64)"},

    {"lc", lc, METH_VARARGS, "Compute the light curve for an array of times using the given parameters."},
    {"lc_loglike", lc_loglike, METH_VARARGS, "Compute the light curve log-likliehood for an array of times using the given parameters."},
    {"bin_data", bin_data, METH_VARARGS, "Bin data with optional linspace for edges."},
    {"median_filter", median_filter, METH_VARARGS, "median filter data given a filter width."},
    {"convolve_1d", convolve_1d, METH_VARARGS, "Convolve filter data given a filter width."},    
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Define the module
static struct PyModuleDef bruce_c_module = {
    PyModuleDef_HEAD_INIT,
    "bruce_c",  // Module name
    NULL,        // Module documentation
    -1,          // Module keeps state in global variables
    bruce_c_methods
};

// Initialize the module
PyMODINIT_FUNC PyInit_bruce_c(void) {
    import_array();  // Initialize the NumPy C API
    return PyModule_Create(&bruce_c_module);
}
