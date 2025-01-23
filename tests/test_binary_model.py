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
                0.2, 1.,
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


t_rv = np.linspace(0,1,100)
rv_model = bruce_c.rv1(t_rv,
                     0,1,
                     10, 0.4 ,np.pi/2,
                     np.pi/2, 0., 0)
rv_err = np.random.uniform(0.1,0.2,t_rv.shape[0])
rv = np.random.normal(rv_model,rv_err)

rv_l = bruce_c.rv1_loglike(t_rv,rv,rv_err,
                     0,1,
                     10, 0.4 ,np.pi/2,
                     np.pi/2, 0., 0,
                     0,0.)




rv1_model, rv2_model =  bruce_c.rv2(t_rv,
                     0,1,
                     5,2, 0.2 ,np.pi/2,
                     np.pi/2, 2., 0)

rv1 = np.random.normal(rv1_model,rv_err)
rv2 = np.random.normal(rv2_model,rv_err)
rv2_l = bruce_c.rv2_loglike(t_rv,rv1,rv2,rv_err,rv_err,
                     0,1,
                     5,2, 0.2 ,np.pi/2,
                     np.pi/2, 2., 0,
                     0,0.)


import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,3, figsize=(12,5))

ax[0].plot(t, lc_arr, c='k', alpha = 0.1)
ax[0].plot(t, model, c='r', lw=1)
ax[0].set(xlabel='Time [day]', ylabel='Flux', title='Loglike = {:.3f}'.format(ll))

ax[1].errorbar(t_rv, rv, yerr=rv_err, fmt='k.', markersize=1, lw=1)
ax[1].plot(t_rv,rv_model, c='r', lw=1)
ax[1].set(xlabel='Phase', ylabel='RV', title='Loglike = {:.3f}'.format(rv_l))



ax[2].errorbar(t_rv, rv1, yerr=rv_err, fmt='r.', markersize=1, lw=1)
ax[2].plot(t_rv, rv1_model, c='r', lw=1)
ax[2].errorbar(t_rv, rv2, yerr=rv_err, fmt='b.', markersize=1, lw=1)
ax[2].plot(t_rv, rv2_model, c='b', lw=1)
ax[2].set(xlabel='Phase', ylabel='RV', title='Loglike = {:.3f}'.format(rv2_l))




plt.tight_layout()
# plt.savefig('../images/binary_model_test.png')
plt.show()

#timeit bruce_c.lc(t, 0., 1.,0.2, 0.2, np.pi/2,0., np.pi/2.,0.7, 0.4,0., 10,0.,2,1)
# import openlc 
# runtime = openlc.context_manager.create_context_and_queue(answers=[0,0])
# timeit openlc.binarystar.lc(t, runtime=runtime)

















