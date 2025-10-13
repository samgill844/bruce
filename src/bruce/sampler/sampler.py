import bruce_c
import  emcee, corner, multiprocess, numpy as np
from datetime import datetime
###############################################################
#                  bruce_c wrappers                           #
###############################################################
# def loglike(y, yerr, model, jitter=0., offset=0):
#     '''
#     Calculate the log-likliehood of the data given the model, errors, jitter, and offset.
#     Assumes y,yerr, and model are arrays of same shape. 
    
#     Parameters:
#         y: array of data points
#         yerr: array of errors
#         model: array of model values
#         jitter: Jitter Value added in quadrature to yerr. 
#         offset: Subtract an arbritrary offset when set to 1.

#     Returns:
#         log-likelihood (double)
#     '''
#     return bruce_c.loglike(y, yerr, model, float(jitter), int(offset))


###############################################################
#                  Other functions                            #
###############################################################


def prior_uniform(val, low, high) : 
    if (val < low) or (val > high) : return -np.inf 
    else : return 0.


def prior_normal(val, reference, reference_err):
    wt = 1. / (reference_err**2)
    return -0.5*((val - reference)**2*wt - np.log(wt))


def prior_bounded_normal(val, reference, reference_err, low, high):
    lp = prior_uniform(val, low, high)
    ln = prior_normal(val, reference, reference_err)
    return ln + lp 



def loglike_(y,yerr, model, jitter=0., offset = True):
    wt = 1. / (yerr**2 + jitter**2)
    if offset : subtracted = np.log(wt)
    else : subtracted = 0.
    return -0.5*((y - model)**2*wt - subtracted)


def loglike(y,yerr, model, jitter=0., offset = True):
    sum = 0.
    for i in range(y.shape[0]):
        ll = loglike_(y[i],yerr[i], model[i], jitter=jitter, offset = offset)
        if (not np.isnan(ll)) and (not np.isinf(ll)) : sum += ll
        #wt = 1. / (yerr[i]**2 + jitter**2)
        #if offset : subtracted = np.log(wt)
        #else : subtracted = 0.
        #sum -= 0.5*((y[i] - model[i])**2*wt - subtracted)
    return sum




def fsfc_to_ew(fs,fc):
    e = fs**2 + fc**2
    w = np.arctan2(fs,fc)
    return e,w


def ew_to_fsfc(e,w):
    fs = np.sqrt(e)*np.sin(w)
    fc = np.sqrt(e)*np.cos(w)
    return fs,fc


def ca_to_h1h2(c, alpha):
    """
    Transform for power-2 law coefficients
    h1 = 1 - c*(1-0.5**alpha)
    h2 = c*0.5**alpha
    :param c: power-2 law coefficient, c
    :param alpha: power-2 law exponent, alpha
    returns: h1, h2
    """
    return 1 - c*(1-0.5**alpha), c*0.5**alpha


def h1h2_to_ca(h1, h2):
    """
    Inverse transform for power-2 law coefficients
    c = 1 - h1 + h2
    alpha = log2(c/h2)
    :param h1: 1 - c*(1-0.5**alpha)
    :param h2: c*0.5**alpha
    returns: c, alpha
    """
    return 1 - h1 + h2, np.log2((1 - h1 + h2)/h2)


def incl_from_radius_1_b(radius_1, b):
    return np.arccos(radius_1*b)



def get_sampler_report(sampler, burn_in, theta_names, verbose=False, name = ''):
    samples = sampler.get_chain(flat=True, discard=burn_in)
    logs = sampler.get_log_prob(flat=True, discard=burn_in) 

    best_idx = np.argmax(logs) 
    best_step = samples[best_idx] 
    low_err = best_step - np.percentile(samples, 16, axis=0)
    high_err = np.percentile(samples, 84, axis=0) - best_step

    return_dict = {}
    best_pars_dict = {}
    return_dict['out_text'] = 'Fit name {:}\nFit completed at {:}\n'.format(name, datetime.now())
    for i in range(len(theta_names)):
        text = '{:>15} = {:.5f} + {:.5f} - {:.5f}'.format(theta_names[i], best_step[i], high_err[i], low_err[i])
        if verbose: print(text)
        return_dict['out_text'] += text + '\n'

        best_pars_dict[theta_names[i]] = [best_step[i], low_err[i], high_err[i]]
    return_dict['pars'] = best_pars_dict
    return_dict['samples'] = samples
    return_dict['samples_raw'] = samples = sampler.get_chain()
    return_dict['logs'] = logs
    return_dict['theta_names'] = theta_names
    return_dict['best_step'] = best_step
    return_dict['burn_in'] = burn_in
    return return_dict

 
def bic(llf, nobs, df_modelwc):
    """
    Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    References
    ----------
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    return -2.0 * llf + np.log(nobs) * df_modelwc


def create_starting_positions(theta, nwalkers, ndim):
    return theta + 1e-4* np.random.randn(nwalkers, ndim)


 
def intilise_and_run_sampler(func, theta, nsteps = 5000, moves=  [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),], nthreads=12, progress=True, refit = True, backend='Backend.h5', args=None):
    ndim = len(theta)
    nwalkers = 2*ndim
    
    if backend is not None:
        backend = emcee.backends.HDFBackend(backend)
        if refit : backend.reset(nwalkers, ndim)


    with multiprocess.Pool(nthreads) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, pool=pool, moves=moves, backend=backend,args=args)
        if refit : 
            p0 = create_starting_positions(theta, nwalkers, ndim)
            sampler.run_mcmc(p0, nsteps, progress=progress, store=True)

    return sampler

def analyse_sampler(sampler, burn_in, theta_names, blob_names = None):
    samples = sampler.get_chain(flat=True, discard=burn_in)
    logs = sampler.get_log_prob(flat=True, discard=burn_in) 

    best_idx = np.argmax(logs) 
    best_step = samples[best_idx] 
    median = np.median(samples,axis=0)
    low_err = median - np.percentile(samples, 16, axis=0)
    high_err = np.percentile(samples, 84, axis=0) - median

    assert best_step.shape[0]==len(theta_names)
    for i in range(len(theta_names)) : 
        print('{:>15} = {:.5f} + {:.5f} - {:.5f}'.format(theta_names[i], best_step[i], high_err[i], low_err[i]))

    if blob_names is not None:
        blobs = sampler.get_blobs(flat=True, discard=burn_in)
        best_blob = blobs[best_idx]
        blob_low_err = best_blob -  np.percentile(blobs, 16, axis=0)
        blob_high_err = np.percentile(blobs, 84, axis=0) - best_blob
        print('Blob auxilery parameters')
        for i in range(len(blob_names)) : 
            print('{:>15} = {:.5f} + {:.5f} - {:.5f}'.format(blob_names[i], best_blob[i], blob_high_err[i], blob_low_err[i]))
    else : blobs, best_blob, blob_low_err, blob_high_err = -1,-1,-1,-1
    return samples, logs, best_idx, best_step, low_err, high_err ,      blobs, best_blob, blob_low_err, blob_high_err

def corner_sampler(sampler, burn_in, theta_names, blob_names=None):
    samples, _, _, best_step, _, _ , _, _, _, _ = analyse_sampler(sampler, burn_in, theta_names=theta_names, blob_names=blob_names)
    fig_corner = corner.corner(samples, labels=theta_names, truths=best_step)
    return fig_corner

def corner_sampler_blobs(sampler, burn_in, theta_names, blob_names):
    _, _, _, _, _, _,  blobs, best_blob, _, _ = analyse_sampler(sampler, burn_in, theta_names=theta_names, blob_names=blob_names)
    fig_corner = corner.corner(blobs, labels=blob_names, truths=best_blob)
    return fig_corner


def get_assymetric_latex_format(vals, dp=5):
    '''
    print the variable with assymetric uncertaintis in latex form
    name : str
    x = [val, low_err, high_err]
    '''
    # First, check for trailing '0' to se if we can reduce dp
    x = [('{:.' + str(dp) + 'f}').format(i) for i in vals]
    while True:
        if dp==0 : 
            x = [('{:.' + str(dp) + 'f}').format(i) for i in vals]
            break
        if (x[0][-1]=='0') & (x[1][-1]=='0') & (x[2][-1]=='0'):
            dp -=1
            x = [('{:.' + str(dp) + 'f}').format(i) for i in vals]
            continue
        break

    x[1] = x[1].lstrip('0').lstrip('.').lstrip('0')
    x[2] = x[2].lstrip('0').lstrip('.').lstrip('0')

    x[1] = x[1].replace('.', '')
    x[2] = x[2].replace('.', '')

    return r'$' + x[0] + '_{(' + x[1] + ')}^{(' + x[2] + ')}' + r'$'

def print_fmt_latex(name, x, dp=5):
    print(name + r' & ' + get_assymetric_latex_format(x, dp=dp) + r'  \\')

def print_vals_from_sampler_latex(theta_names, best_step, low_error, high_error, dp=10):
    for i in range(len(theta_names)):
        print_fmt_latex(theta_names[i], [best_step[i], low_error[i], high_error[i]], dp=dp)
