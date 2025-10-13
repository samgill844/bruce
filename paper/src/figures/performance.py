# Imports 
import numpy as np, bruce
np.random.seed(0)
import matplotlib.pyplot as plt 
from astropy.time import Time
from tqdm import tqdm
import glob
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})

REFIT=False

def template_match_lightcurve(time, flux, flux_err, normalisation_model):
    start = Time.now()
    for _ in range(10) : bruce.template_match.template_match_lightcurve(time, flux, flux_err, normalisation_model)
    end = Time.now()
    return  86400*(end-start).jd/3

log_2 = np.arange(0, 21)
runtime_name = '2.6 GHz 6-Core Intel Core i7'
noise = 1e-3





if REFIT:
    with open(runtime_name+'.txt', 'w+') as f:
        for i in tqdm(range(log_2.shape[0])):
            x = np.linspace(0,1, 2**log_2[i])
            ye = np.random.uniform(0.5*noise,1.5*noise,x.shape[0])
            y = np.random.normal(np.ones(x.shape), ye)
            normalisation_model = np.ones(x.shape)
            
            time = template_match_lightcurve(x,y,ye, normalisation_model)
            f.write('{:},{:}\n'.format(int(2**log_2[i]), time))
else:
    files = glob.glob('*.txt')
    fig, ax = plt.subplots(1,1)
    for file in files:
        name = file.split('.')[0]
        n, time = np.loadtxt(file, delimiter=',').T
        plt.plot(n, time, label=name)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('N')
    plt.ylabel('time (s)')
    plt.legend()
    plt.show()