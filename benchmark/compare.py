import numpy as np 
from bruce.binarystar import lc 
import time
import cpuinfo 
import nvgpu
import numba.cuda
import matplotlib.pyplot as plt 
import pickle 
import sys 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

if __name__=='__main__':
    numb_of_pickles = len(sys.argv) -1 
    
    if numb_of_pickles==0 : exit() 


    # Make the plot 
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(7.48031,3))
    ax1 = plt.gca()
    ax1.grid(which='major')
    ax1.grid(which='minor', alpha = 0.2)
    ax1.set_ylabel('$\mathcal{L}$ / s')
    ax1.set_xlabel('Data size')

    fig2 = plt.figure(figsize=(7.48031,3))
    ax2 = plt.gca()

    for k in range(numb_of_pickles):
        # load th epickle
        test_results = load_obj( sys.argv[k+1] )
        keys = [i for  i in test_results.keys()]
        for i in range(len(test_results)):
            x = np.array([2**j[0] for j in test_results[keys[i]]])
            y = np.array([j[2] for j in test_results[keys[i]]])
            ax1.loglog(x,y, label=keys[i], ls='--') 

            if i > 0:
                x = np.array([2**j[0] for j in test_results[keys[0]]])
                y = np.array([j[2] for j in test_results[keys[i]]]) / np.array([j[2] for j in test_results[keys[0]]])
                ax2.semilogx(x,y, label=keys[i], ls='--') 


    ax2.grid(which='major')
    ax2.grid(which='minor', alpha = 0.2)
    ax2.set_ylabel('$\mathcal{L}$ / s')
    ax2.set_xlabel('Data size')

    ax1.legend()

    fig.subplots_adjust(left = 0.08)
    fig.subplots_adjust(bottom = 0.15)
    fig.tight_layout()

    fig2.subplots_adjust(left = 0.08)
    fig2.subplots_adjust(bottom = 0.15)
    fig2.tight_layout()
    fig.savefig('scale_test.png')

    plt.show()






