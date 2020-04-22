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


    for k in range(numb_of_pickles):
        # load th epickle
        test_results = load_obj( sys.argv[k+1] )
        keys = [i for  i in test_results.keys()]
        for i in range(len(test_results)):
            x = np.array([2**j[0] for j in test_results[keys[i]]])
            y = np.array([j[2] for j in test_results[keys[i]]])
            plt.loglog(x,y, label=keys[i], ls='--') 



    plt.legend()
    plt.tight_layout()
    plt.grid(which='major')
    plt.grid(which='minor', alpha = 0.2)
    plt.ylabel('$\mathcal{L}$ / s')
    plt.xlabel('Data size')

    plt.subplots_adjust(left = 0.08)
    plt.subplots_adjust(bottom = 0.15)

    plt.savefig('scale_test.png')
    plt.show()


