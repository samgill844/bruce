import bruce_c

def loglike(y, yerr, model, jitter=0., offset=0):
    '''
    Calculate the log-likliehood of the data given the model, errors, jitter, and offset.
    Assumes y,yerr, and model are arrays of same shape. 
    
    Parameters:
        y: array of data points
        yerr: array of errors
        model: array of model values
        jitter: Jitter Value added in quadrature to yerr. 
        offset: Subtract an arbritrary offset when set to 1.

    Returns:
        log-likelihood (double)
    '''
    return bruce_c.loglike(y, yerr, model, float(jitter), int(offset))
