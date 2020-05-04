import numpy as np 
from bruce.binarystar import lc 
import time
import cpuinfo 
import nvgpu
import numba.cuda
import matplotlib.pyplot as plt 
import pickle 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Get device information
processor = cpuinfo.get_cpu_info()['brand'] 
gpu_info = nvgpu.gpu_info()
test_results = dict()



# N dyadic
N = np.arange(3,22)

tstart = 0
tend = 80
period = 8.12
t_zero = 4.215
radius_1 = 0.14
k = 0.05


print('CPU test') 
print(processor)
test_results[processor] = []

for i in range(N.shape[0]):
    t = np.linspace(tstart,tend, 2**N[i], dtype = np.float64)
    m = -2.5*np.log10(lc(t, t_zero=t_zero, period=period, radius_1=radius_1, k=k)).astype(np.float64)
    me = np.random.uniform(0.01,0.02,t.shape[0]).astype(np.float64)
    m = np.random.normal(m,me)

    # Make a first call 
    L = lc(t,m,me, t_zero=t_zero, period=period, radius_1=radius_1, k=k)

    # now it's compiled, test
    start = time.time() 
    for j in range(100) : L = lc(t,m,me, t_zero=t_zero, period=period, radius_1=radius_1, k=k)
    end = time.time() 
    tpercall = (end-start)/100.
    print('{:>5} {:.6f} {:,}'.format(N[i], tpercall, int(1./tpercall)))
    test_results[processor].append([N[i], tpercall, 1./tpercall])


# Now get GPU info
print('\nNumber of GPUs: {:}'.format(len(gpu_info)))
if len(gpu_info) > 0:
    for k in range(len(gpu_info)):
        print('ID : {:} -> GPU: {:}'.format(gpu_info[k]['index'],  gpu_info[k]['type'] ))
        test_results[gpu_info[k]['type']] = []

        # Set device context
        numba.cuda.select_device(k)

        for i in range(N.shape[0]):
            t = np.linspace(tstart,tend, 2**N[i], dtype = np.float64)
            m = -2.5*np.log10(lc(t, t_zero=t_zero, period=period, radius_1=radius_1, k=k)).astype(np.float64)
            me = np.random.uniform(0.01,0.02,t.shape[0]).astype(np.float64)
            m = np.random.normal(m,me)

            t_ = numba.cuda.to_device(t)
            m_ = numba.cuda.to_device(m)
            me_ = numba.cuda.to_device(me)
            L_ = numba.cuda.to_device(np.zeros(t.shape[0], dtype = np.float64))
            threads_per_block = 256
            blocks = int(np.ceil(t.shape[0]/threads_per_block))

            # Make the first call 
            Lgpu = lc(t_,m_,me_,  gpu=1, loglike=L_, blocks = blocks, threads_per_block = threads_per_block)

            # now it's compiled, test
            start = time.time() 
            for j in range(100) : Lgpu = lc(t_,m_,me_,  gpu=1, loglike=L_, blocks = blocks, threads_per_block = threads_per_block)
            end = time.time() 
            tpercall = (end-start)/100.
            print('{:>5} {:.6f} {:,}'.format(N[i], tpercall, int(1./tpercall)))
            test_results[gpu_info[k]['type']].append([N[i], tpercall, 1./tpercall])


# Now dave pickle name of processor
save_obj(test_results, processor )

# Make the plot 
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(7.48031,3))

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
