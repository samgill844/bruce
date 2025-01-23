import numpy as np
import bruce_c

N = int(100000)
t = np.linspace(-0.2,0.2, N).astype(np.float64)
t32 = t.astype(int)

timeit bruce_c.lc(t, 0., 1.,0.2, 0.2, np.pi/2,0., np.pi/2.,0.7, 0.4,0.,10,0.,2,0)
timeit bruce_c.lc(t32, 0., 1.,0.2, 0.2, np.pi/2,0., np.pi/2.,0.7, 0.4,0.,10,0.,2,0)