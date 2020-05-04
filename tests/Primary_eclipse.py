import numpy as np 
import matplotlib.pyplot as plt
from bruce.binarystar import lc 
from ellc import lc as elc 
plt.rcParams["font.family"] = "Times New Roman"


radius_1 = 0.2 
k = 0.2 
radius_2 = radius_1*k 
incl = 89.
ldc_1_1 = 0.8
ldc_1_2 = 0.8
ld_law_1 = 2 # power-2
t = np.linspace(-0.05,0.05,1000)

F_bruce = lc(t, radius_1, k=k, incl=incl, ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_2, ld_law_1=ld_law_1)
F_ellc = elc(t, radius_1, radius_2, 0., incl, ld_1 = 'power-2', ldc_1=ldc_1_1, ldc_2=ldc_1_2)

f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize = (5,7))
ax1.plot(t,F_bruce, 'b', linewidth = 0.25 )
ax1.plot(t,F_ellc, 'r' , linewidth = 0.25 )

ax2.plot(t, 1e6*(1 - F_bruce/F_ellc), linewidth = 0.25 )


ax1.set_title('Primary eclipse [power-2]')
ax1.grid() 
ax2.grid() 
ax1.set_ylabel('Flux')
ax1.set_xlabel('Phase')
ax2.set_ylabel('Noise [ppm]')
ax2.set_xlabel('Phase')

plt.tight_layout()
plt.show()