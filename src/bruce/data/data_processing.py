import bruce_c

def median_filter(time, flux, bin_size=0.5/24/3) : return bruce_c.median_filter(time, flux, bin_size)
def convolve_1d(time, flux, bin_size=0.5/24/3) : return bruce_c.convolve_1d(time, flux, bin_size)
def bin_data(time, flux, bin_size=0.5/24/3) : return bruce_c.bin_data(time, flux, bin_size)

