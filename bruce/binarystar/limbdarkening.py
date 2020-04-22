import math, numba

@numba.njit
def limb_darkening_intensity(ld_law, ldc_1_1, ldc_1_2,ldc_1_3, ldc_1_4,   ri):
    # 0 = uniform
    # 1 = quadratic
    # 2 = power-2
    u = math.sqrt(1 - ri**2)

    

    if (ld_law==2) :
        I_0 = (ldc_1_2+2)/(math.pi*(ldc_1_2-ldc_1_1*ldc_1_2+2))
        return I_0*(1 - ldc_1_1*(1 - u**ldc_1_2))