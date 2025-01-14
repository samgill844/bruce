import numpy as np
import bruce_c

N = int(1e6)
y = np.random.normal(1,1e-3,N).astype(np.float64)
yerr = np.random.uniform(1e-3,1e-2,N).astype(np.float64)
model = np.ones(N, dtype=np.float64)
jitter=0
offset=0
ll = bruce_c.loglike(y,yerr,model, jitter, offset)
print('Loglike = {:}'.format(ll))


N = int(100000)
t = np.linspace(-0.2,0.2, N).astype(np.float64)
lc_arr = bruce_c.lc(t, 
                0., 1.,
        0.2, 0.2, np.pi/2,
        0., np.pi/2.,
        0.7, 0.4,
        0., 10,
        0.,
        2,
        1)
model = lc_arr.copy()
lc_err = np.random.uniform(1e-3,1e-2,N)
lc_arr = np.random.normal(lc_arr, lc_err)
ll = bruce_c.lc_loglike(t,lc_arr, lc_err, 
                0., 1.,
        0.2, 0.2, np.pi/2,
        0., np.pi/2.,
        0.7, 0.4,
        0., 10,
        0.,
        2,
        1,
        0.,0)

import matplotlib.pyplot as plt
plt.plot(t, lc_arr, c='k', alpha = 0.1)
plt.plot(t, model, c='r', lw=1)

plt.title('Loglike = {:}'.format(ll))
plt.show()

#timeit bruce_c.lc(t, 0., 1.,0.2, 0.2, np.pi/2,0., np.pi/2.,0.7, 0.4,0., 10,0.,2,1)
# import openlc 
# runtime = openlc.context_manager.create_context_and_queue(answers=[0,0])
# timeit openlc.binarystar.lc(t, runtime=runtime)

















