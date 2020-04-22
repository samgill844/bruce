import numba, math 


@numba.njit
def Fellipsoidal(nu, q, radius_1, incl, u, y):
    # SUrray of https:#books.google.co.uk/books?id=ngtmDwAAQBAJ&pg=PA239&lpg=PA239&dq=ellipsoidal+variation+approximation+binary+star&source=bl&ots=swiO_JQdIR&sig=ACfU3U0HVtS8G37Z7EbdjDymUqICD36FgA&hl=en&sa=X&ved=2ahUKEwiO1tH9ud7hAhWDaFAKHRVoASIQ6AEwC3oECAkQAQ#v=onepage&q=ellipsoidal%20variation%20approximation%20binary%20star&f=false
    # q - mass ratio
    # radius_1 - R*/a 
    # incl - orbital inclination 
    # u - linear limb-darkening coefficient
    # y - gravitational darkening coefficient 
    # nu - true anomaly 
    #nu /= 2
    # true anomaly goes from -pi to pi 
    # to convert to phase,we could ass pi and divide by two pi
    alpha1 = ((y+2.)/(y+1.))*25.*u / (24.*(15. + u))
    alpha2 = (y+1.)*(3.*(15.+u))/(20.*(3.-u))
    Ae = alpha2*q*math.pow(radius_1,3)*math.pow(math.sin(incl),2) # Amplitude of ellipsoidal variation

    # Harmonic coeefs 
    f1 = 3*alpha1*radius_1*(5*math.pow(math.sin(incl),2) - 4)/math.sin(incl)
    f2 = 5*alpha1*radius_1*math.sin(incl)

    # Now return the variation 
    return -Ae*( math.cos(2*nu)    +    f1*math.cos(nu)      +      f2*math.cos(3*nu) ) 

'''
import numpy as np 
import matplotlib.pyplot as plt 

nu = np.linspace(0, 2*np.pi, 100) 
q = 0.4 
radius_1 = 0.2 
k = 0.2 
incl = math.pi/2 
u = 0.5
y = 0.4 

for i in nu : plt.scatter(i, Fellipsoidal(i, q, radius_1, incl, u, y) ) 
plt.show()
'''