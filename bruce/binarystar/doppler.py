import numba, math

@numba.njit
def Fdoppler(phase, A_doppler):
    # https://arxiv.org/pdf/1305.3271.pdf
    #return A_doppler*math.cos(phase+math.pi)
    return A_doppler*math.cos(phase)