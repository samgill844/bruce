import numba, numpy as np , math 


###################################################
# Fortran conversions
###################################################
@numba.njit
def sign(a,b) : 
    if b >= 0.0 : return abs(a)
    return -abs(a)

@numba.njit
def merge(a,b,mask):
    if mask : return a 
    else : return b

###################################################
# Brent minimisation
###################################################

@numba.njit
def brent(ax, bx, cx, e, incl, w, radius_1):
    #######################################################
    # Find the minimum of a function (func) between ax and
    # cx and that func(bx) is less than both func(ax) and 
    # func(cx). 
    # z0 is the additional arguments 
    #######################################################
    # pars
    tol = 1e-5
    itmax = 100
    eps = 1e-5
    cgold = 0.3819660
    zeps = 1.0e-10

    a = min(ax, cx)
    b = max(ax, cx)
    d = 0.
    v = bx
    w = v
    x = v
    e = 0.0
    #fx = func(x,z0)
    fx = get_z(x, e, incl, w, radius_1)
    fv = fx
    fw = fx
    #print(ax,bx,cx)

    for iter in range(itmax):
        xm = 0.5*(a+b)
        tol1 = tol*abs(x)+zeps
        tol2 = 2.0*tol1
        #print(iter, x, abs(x-xm), tol2-0.5*(b-a))
        if(abs(x-xm) <= (tol2-0.5*(b-a))):
            return x
        
        if(abs(e) > tol1):
            r = (x-w)*(fx-fv)
            q = (x-v)*(fx-fw)
            p = (x-v)*q - (x-w)*r
            q = 2.0*(q-r)
            if (q > 0.0) :  p = - p
            q = abs(q)
            etemp = e
            e = d

            if (  (abs(p) >= abs(.5*q*etemp)) or (p <= q*(a-x)) or (p >= q*(b-x))):
                e = merge(a-x, b-x, p >= q*(b-x))
                d = cgold*e
            else:
                d = p/q
                u=x+d
                if ( ((u-a) < tol2) or ((b-u) < tol2)) :  d = sign(tol1, xm-x)
        else:
            e = merge(a-x, b-x, x >= xm)
            d = cgold*e

        u = merge(x+d, x+sign(tol1,d), abs(d) >= tol1)
        #fu = func(u,z0)
        fu = get_z(u, e, incl, w, radius_1)


        if (fu <= fx):
            if ( u >= x) : a = x 
            else : b = x 
            #shft(v,w,x,u)
            v = w 
            w = x 
            x = u 
            #shft(fv,fw,fx,fu)
            fv = fw
            fw = fx 
            fx = fu 
        else:
            if (u < x) : a = u 
            else : b = u 
            if ((fu <= fw) or (w==x)):
                v=w
                fv=fw
                w=u
                fw=fu
            elif ((fu <= fv) or (v==x) or (v==w)):
                v = u
                fv = fu 
    return 1.


# Gravatational constant 
G = 6.67408e-11

#######################################
#          Keplers quation            #
#######################################

@numba.njit(fastmath=True)
def kepler(M, E, e) : return M - E - e*math.sin(E) 

@numba.njit(fastmath=True)
def kdepler(E, e) : return -1 + e*math.cos(E) 


@numba.njit(fastmath=True)
def fmod(a, b) : return a % abs(b)

#######################################
#        Eccentric anomaly            #
#######################################
@numba.njit(fastmath=True)
def getEccentricAnomaly(M, ecc, E_tol):
    '''
    m =  fmod(M , 2*math.pi) # should be fmod
    f1 = fmod(m , (2*math.pi) + e*math.sin(m)) + e*e*math.sin(2.0*m)/2.0  # should be fmod around 
    test  =1.0 
    e1 = 0.
    e0 = 1.0 
    while (test > E_tol):
        e0 = e1 
        e1 = e0 + (m-(e0 - e*math.sin(e0)))/(1.0 - e*math.cos(e0))
        test = abs(e1-e0)

    if (e1 < 0) : e1 = e1 + 2*math.pi 
    return e1 
    '''
    M = fmod(M , 2*math.pi)
    if ecc == 0 : return M
    if M > math.pi:
        M = 2*math.pi - M
        flip = True
    else:
        flip = False
    alpha = (3*math.pi + 1.6*(math.pi-abs(M))/(1+ecc) )/(math.pi - 6/math.pi)
    d = 3*(1 - ecc) + alpha*ecc
    r = 3*alpha*d * (d-1+ecc)*M + M**3
    q = 2*alpha*d*(1-ecc) - M**2
    w = (abs(r) + math.sqrt(q**3 + r**2))**(2/3)
    E = (2*r*w/(w**2 + w*q + q**2) + M) / d
    f_0 = E - ecc*math.sin(E) - M
    f_1 = 1 - ecc*math.cos(E)
    f_2 = ecc*math.sin(E)
    f_3 = 1-f_1
    d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
    d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3**2)*f_3/6)
    E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4**2*f_3/6 - d_4**3*f_2/24)
    if flip : E =  2*math.pi - E
    return E
 
#######################################
#      Time of periastron passage     #
#######################################
@numba.njit(fastmath=True)
def t_ecl_to_peri(t_ecl, e, w, incl, p_sid,radius_1):
    
    # Define variables used
    efac  = 1.0 - e*2       # check if 2 or power
    sin2i = math.sin(incl)**2.

    # Value of theta for i=90 degrees
    theta_0 = (math.pi/2.) - w       # True anomaly at superior conjunction

    # Accurate time of eclipse prior to periastro 
    '''
    if incl != math.pi/2 :
        theta = brent(theta_0 - math.pi/2, theta_0, theta_0 + math.pi/2,  e, incl, w, radius_1)
        print(theta_0, theta)
    else:
        theta = theta_0
    '''
    #if theta < 0 : theta = 1 + theta

    # Instead, we assume that the time immediately prior to periastron is
    # at superiod conjunction is good. 
    theta = theta_0 # BODGE!

    if (theta == math.pi) :  ee = math.pi
    else :  
        ee =  2.0 * math.atan(math.sqrt((1.-e)/(1.0+e)) * math.tan(theta/2.0))

    eta = ee - e*math.sin(ee)
    delta_t = eta*p_sid/(2*math.pi)
    return t_ecl  - delta_t 

#######################################
#      Calculate the true anomaly     #
#######################################
@numba.njit(fastmath=True)
def getTrueAnomaly(time, e, w, period,  t_zero, incl, E_tol, radius_1 ):
    # Get time immediatly priore to periastron passage
    tp = t_ecl_to_peri(t_zero, e, w, incl, period, radius_1)  

    if e<1e-5:
        return ((time - tp)/period - math.floor(((time - tp)/period)))*2.*math.pi
    else:
        # Calcualte the mean anomaly
        M = 2*math.pi*((time -  tp  )/period % 1.)

        # Calculate the eccentric anomaly
        E = getEccentricAnomaly(M, e, E_tol)

        # Now return the true anomaly
        return 2.*math.atan(math.sqrt((1.+e)/(1.-e))*math.tan(E/2.)) 

#######################################
#  Calculate the projected seperaton  #
#######################################
@numba.njit(fastmath=True)
def get_z(nu, e, incl, w, radius_1) : return (1-e**2) * math.sqrt( 1.0 - math.sin(incl)**2  *  math.sin(nu + w)**2) / (1 + e*math.cos(nu)) /radius_1

#@numba.njit(fastmath=True)
#def get_z_(nu, z) : return get_z(nu, z[0], z[1], z[2], z[3]) # for brent

#######################################
#  Calculate the projected seperaton  #
#######################################
@numba.njit(fastmath=True)
def getProjectedPosition(nu, w, incl) : return math.sin(nu + w)*math.sin(incl) 


#######################################
#     Calculate the mass function     #
#######################################
@numba.njit(fastmath=True)
def mass_function_1(e, P, K1) : return (1-e*e) **1.5*P*86400.1* (K1*1000)**3/(2*math.pi*G*1.989e30)

@numba.njit(fastmath=True)
def mass_function_1_(M2, z) : return ((M2*math.sin(z[1]))**3 / ( (z[0] + M2)**2)) - mass_function_1(z[2], z[3], z[4]) 

@numba.njit(fastmath=True)
def z_func_(time, Z, e, w, period,  t_zero, incl, E_tol, radius_1):
    for i in range(time.shape[0]):
        Z[i] = getTrueAnomaly(time[i], e, w, period,  t_zero, incl, E_tol, radius_1 )
        Z[i] = get_z(Z[i], e, incl, w, radius_1)


def z_func(time, e=0, w=np.pi/2, period=1,  t_zero=0, incl=np.pi/2, E_tol=1e-5, radius_1=0.2):
    Z = np.empty_like(time)
    z_func_(time, Z, e, w, period,  t_zero, incl, E_tol, radius_1)
    return Z


import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_secondary_phase(fs, fc, incl, radius_1=0.2):
    t = np.linspace(0,1,10000)
    e = fs**2 + fc**2
    w = np.arctan2(fs,fc)
    z = z_func(t, e=e, w=w, period=1,  t_zero=0, incl=incl, E_tol=1e-5, radius_1=radius_1)


    #plt.plot(t, z)
    peaks, _ = find_peaks(-z[500:9500])
    #if len(peaks)==3 : peaks = peaks[1] 
    
    return t[500:9500][peaks][0]
    #plt.axvline(t[peaks])
    #plt.show()
