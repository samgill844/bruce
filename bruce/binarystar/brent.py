import numba 


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
def brent(func,x1,x2, z0):
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