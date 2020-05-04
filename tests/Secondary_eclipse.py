import numpy as np 
import matplotlib.pyplot as plt
from bruce.binarystar import lc 
from bruce.binarystar.kepler import find_secondary_phase
from ellc import lc as elc 
plt.rcParams["font.family"] = "Times New Roman"


radius_1 = 0.2 
k = 0.2 
radius_2 = radius_1*k 
incl = 89.
ldc_1_1 = 0.8
ldc_1_2 = 0.8
ld_law_1 = 2 # power-2
t = np.linspace(-0.05,0.9,10000) 
sbratio = 0.1
fs = 0.2
fc = 0.4

F_bruce = lc(t, radius_1, k=k, incl=incl, ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_2, ld_law_1=ld_law_1, SBR=sbratio, fs=fs, fc=fc)
F_ellc = elc(t, radius_1, radius_2, sbratio, incl, ld_1 = 'power-2', ldc_1=ldc_1_1, ldc_2=ldc_1_2, f_s = fs, f_c = fc)

f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,7))


ax1.plot(t,F_bruce, 'b', linewidth = 0.25 )
ax1.plot(t,F_ellc, 'r' , linewidth = 0.25 )
ax3.plot(t,F_bruce, 'b', linewidth = 0.25 )
ax3.plot(t,F_ellc, 'r' , linewidth = 0.25 )

ax2.plot(t, 1e6*(1 - F_bruce/F_ellc), linewidth = 0.25 )
ax4.plot(t, 1e6*(1 - F_bruce/F_ellc), linewidth = 0.25 )


ax1.set_xlim(-0.05,0.05)
ax2.set_xlim(-0.05,0.05)

sphase = find_secondary_phase(fs,fc)
ax3.set_xlim(sphase - 0.05, sphase + 0.05)
ax4.set_xlim(sphase - 0.05, sphase + 0.05)

ax1.set_title('Primary eclipse [power-2]')
ax3.set_title('Secondary eclipse [power-2]')


ax1.set_ylabel('Flux')
ax1.set_xlabel('Phase')
ax2.set_ylabel('Noise [ppm]')
ax2.set_xlabel('Phase')

ax3.set_ylabel('Flux')
ax3.set_xlabel('Phase')
ax4.set_ylabel('Noise [ppm]')
ax4.set_xlabel('Phase')





plt.tight_layout()
plt.show()