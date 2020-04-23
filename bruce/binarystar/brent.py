import numba 
import matplotlib.pyplot as plt
import numpy as np 

###################################################
# Fortran conversions
###################################################
@numba.njit
def sign(a,b) : 
    if b >= 0.0 : return abs(a)
    return -abs(a)

    

###################################################
# Brent minimisation
###################################################

@numba.njit
def zbrent(func,x1,x2, z0):
    #######################################################
    # Find the root of a function (func) between x1 and x2.
    # z0 is the additional arguments 
    #######################################################

    # pars
    tol = 1e-5
    itmax = 100
    eps = 1e-5

    a = x1
    b = x2
    c = 0.
    d = 0.
    e = 0.
    fa = func(a,z0)
    fb = func(b,z0)

    fc = fb

    for iter in range(itmax):
        if (fb*fc > 0.0):
            c = a
            fc = fa
            d = b-a
            e=d   

        if (abs(fc) < abs(fb)):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0*eps*abs(b)+0.5*tol
        xm = (c-b)/2.0
        if (abs(xm) <  tol1 or fb == 0.0) : return b

        if (abs(e) > tol1 and abs(fa) >  abs(fb)):
            s = fb/fa
            if (a == c):
                p = 2.0*xm*s
                q = 1.0-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            
            if (p > 0.0) : q = - q
            p = abs(p)
            if (2.0*p < min(3.0*xm*q-abs(tol1*q),abs(e*q))):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d   

        a = b
        fa = fb      
         
        if( abs(d) > tol1) : b = b + d
        else : b = b + sign(tol1, xm)

        fb = func(b,z0)
    return 1


@numba.njit
def merge(a,b,mask):
    if mask : return a 
    else : return b

@numba.njit
def brent(func,ax, bx, cx, z0):
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
    fx = func(x,z0)
    fv = fx
    fw = fx

    for iter in range(itmax):
        xm = 0.5*(a+b)
        tol1 = tol*abs(x)+zeps
        tol2 = 2.0*tol1
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
        fu = func(u,z0)


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


if __name__=='__main__':
    x = np.linspace(-1,1,100)
    f, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize = (5,7))
    ######################################
    # zbrent test to find root finding
    ######################################
    @numba.njit
    def func(x, y) : return -0.5*x + 0.1 

    y = func(x, 0)
    ax1.plot(x,y, c='k')
    ax1.axvline(-0.5, ls='--', alpha = 0.5)
    ax1.axvline(0.5, ls='--', alpha = 0.5)
    ax1.axhline(0., ls='--', alpha = 0.5)
    xo = zbrent(func,-0.5,0.5, (0,))
    ax1.plot(xo,0, 'r*')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('z-brent root find')

    @numba.njit
    def func(x, y) : return 0.5*x**2 + 0.1 

    y = func(x, 0)

    ax2.plot(x,y, c='k')
    xo = brent(func,-0.5, 0.1, 0.5, (0,))
    ax2.axvline(0.1, ls='--', alpha = 0.5)
    ax2.axvline(xo, ls='--', alpha = 1.)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('brent minimisation')
    print(xo)
    plt.tight_layout()
    plt.show()
