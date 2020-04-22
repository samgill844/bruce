import math, numba 


@numba.njit 
def clip(a, b, c) : 
    if a < b : return b 
    if a > c : return c 
    return a

@numba.njit
def ellpic_bulirsch( n,  k):
    kc = math.sqrt(1.-k*k)
    p = math.sqrt(n + 1.)
    m0 = 1.
    c = 1.
    d = 1./p
    e = kc
    nit = 0
    while(nit < 10000):
        f = c
        c = d/p + c
        g = e/p
        d = 2.*(f*g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if(math.fabs(1.-kc/g) > 1.0e-8):
            kc = 2.*math.sqrt(e)
            e = kc*m0
        else : return 0.5*math.pi*(c*m0+d)/(m0*(m0+p))
        nit+=1
    return 0.

@numba.njit
def ellec(k):
    # Computes polynomial approximation for the complete elliptic
    # integral of the first kind (Hasting's approximation):
    m1 = 1.0 - k*k
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    ee1 = 1.0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ee2 = m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))*math.log(1.0/m1)
    ellec = ee1 + ee2
    return ellec

@numba.njit
def ellk( k):
    # Computes polynomial approximation for the complete elliptic
    # integral of the second kind (Hasting's approximation):
    m1 = 1.0 - k*k
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012
    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4))))*math.log(m1)
    ellk = ek1 - ek2
    return ellk



@numba.njit
def Flux_drop_analytical_quadratic( d,  p,  c1,  c2,  tol):
    '''
    Calculate the analytical flux drop from the quadratic limb-darkening law.

    Parameters
    d : double
        Projected seperation of centers in units of stellar radii.
    p : double
        Ratio of the radii.
    c : double
        The first power-2 coefficient.
    a : double
        The second power-2 coefficient.
    f : double
        The flux from which to drop light from.
    eps : double
        Factor (1e-9)
    '''

    kap0 = 0.0
    kap1 = 0.0
    omega = 1.0 - c1/3.0 - c2/6.0

    # allow for negative impact parameters
    d = math.fabs(d)

    # check the corner cases
    if(math.fabs(p - d) < tol) : d = p
    
    if(math.fabs(p - 1.0 - d) < tol) : d = p - 1.0
    
    if(math.fabs(1.0 - p - d) < tol) : d = 1.0 - p
    
    if(d < tol) : d = 0.0
    

    x1 = math.pow((p - d), 2.0)
    x2 = math.pow((p + d), 2.0)
    x3 = p*p - d*d

    #source is unocculted:
    if(d >= 1.0 + p) : return 0.
    
    #source is completely occulted:
    if(p >= 1.0) and (d <= p - 1.0):
        lambdad = 0.0
        etad = 0.5        #error in Fortran code corrected here, following Jason Eastman's python code
        lambdae = 1.0
        return - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*(lambdad + 2.0/3.0) + c2*etad)/omega
    
    #source is partly occulted and occulting object crosses the limb:
    if(d >= math.fabs(1.0 - p)) and (d <= 1.0 + p):
        kap1 = math.acos(   clip(min((1.0 - p*p + d*d)/2.0/d, 1.0),-1,1)   )
        #print(p, d, (p*p + d*d - 1.0)/2.0/p/d )
        kap0 = math.acos(   clip(min((p*p + d*d - 1.0)/2.0/p/d, 1.0),-1,1) )
        lambdae = p*p*kap0 + kap1
        lambdae = (lambdae - 0.50*math.sqrt(max(4.0*d*d - math.pow((1.0 + d*d - p*p), 2.0), 0.0)))/math.pi
    

    #edge of the occulting star lies at the origin
    if(d == p):
        if(d < 0.5):
            q = 2.0*p
            Kk = ellk(q)
            Ek = ellec(q)
            lambdad = 1.0/3.0 + 2.0/9.0/math.pi*(4.0*(2.0*p*p - 1.0)*Ek + (1.0 - 4.0*p*p)*Kk)
            etad = p*p/2.0*(p*p + 2.0*d*d)
            return - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega

        elif(d > 0.5):
            q = 0.5/p
            Kk = ellk(q)
            Ek = ellec(q)
            lambdad = 1.0/3.0 + 16.0*p/9.0/math.pi*(2.0*p*p - 1.0)*Ek -  \
                (32.0*math.pow(p, 4.0) - 20.0*p*p + 3.0)/9.0/math.pi/p*Kk
            etad = 1.0/2.0/math.pi*(kap1 + p*p*(p*p + 2.0*d*d)*kap0 -  \
                (1.0 + 5.0*p*p + d*d)/4.0*math.sqrt((1.0 - x1)*(x2 - 1.0)))

        else:
            lambdad = 1.0/3.0 - 4.0/math.pi/9.0
            etad = 3.0/32.0
            return - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega
        

        return - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega
    
    #occulting star partly occults the source and crosses the limb:
    #if((d > 0.5 + math.fabs(p  - 0.5) && d < 1.0 + p) || (p > 0.5 && d > math.fabs(1.0 - p)*1.0001 \
    #&& d < p))  #the factor of 1.0001 is from the Mandel/Agol Fortran routine, but gave bad output for d near math.fabs(1-p)
    if ( (d > 0.5 + math.fabs(p  - 0.5)) and (d < 1.0 + p) ) or ( (p > 0.5) and (d > math.fabs(1.0 - p)) and (d < p)):
        q = math.sqrt((1.0 - x1)/4.0/d/p)
        Kk = ellk(q)
        Ek = ellec(q)
        n = 1.0/x1 - 1.0
        Pk = ellpic_bulirsch(n, q)
        lambdad = 1.0/9.0/math.pi/math.sqrt(p*d)*(((1.0 - x2)*(2.0*x2 +  \
                x1 - 3.0) - 3.0*x3*(x2 - 2.0))*Kk + 4.0*p*d*(d*d +  \
                7.0*p*p - 4.0)*Ek - 3.0*x3/x1*Pk)
        if(d < p) : lambdad += 2.0/3.0
        etad = 1.0/2.0/math.pi*(kap1 + p*p*(p*p + 2.0*d*d)*kap0 -  \
            (1.0 + 5.0*p*p + d*d)/4.0*math.sqrt((1.0 - x1)*(x2 - 1.0)))
        return - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega
    
    #occulting star transits the source:
    if (p <= 1.0) and (d <= (1.0 - p)):
        etad = p*p/2.0*(p*p + 2.0*d*d)
        lambdae = p*p

        q = math.sqrt((x2 - x1)/(1.0 - x1))
        Kk = ellk(q)
        Ek = ellec(q)
        n = x2/x1 - 1.0
        Pk = ellpic_bulirsch(n, q)

        lambdad = 2.0/9.0/math.pi/math.sqrt(1.0 - x1)*((1.0 - 5.0*d*d + p*p +  \
                x3*x3)*Kk + (1.0 - x1)*(d*d + 7.0*p*p - 4.0)*Ek - 3.0*x3/x1*Pk)

        # edge of planet hits edge of star
        if(math.fabs(p + d - 1.0) <= tol):        
            lambdad = 2.0/3.0/math.pi*math.acos( clip(1.0 - 2.0*p,-1,1 )) - 4.0/9.0/math.pi* \
                        math.sqrt(p*(1.0 - p))*(3.0 + 2.0*p - 8.0*p*p)
        
        if(d < p) : lambdad += 2.0/3.0
    
    return - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega


'''
import numpy as np 
import matplotlib.pyplot as plt 
z = np.linspace(-2,2,1000)
z_ = np.abs(z) 

k = 0.1 
c1 = 0.5
c2 = 0.2
tol = 1e-8  


for i in range(len(z)) : plt.scatter(z[i], Flux_drop_analytical_quadratic( z_[i],  k,  c1,  c2,  tol), c='k', s=10)
plt.show() 
'''