import numpy as np
from scipy import optimize
from scipy import special
import cmath as cm
import mpmath as mp

hbar = 1

sigma_sq = 1
E_c = 1

def R_0(hbarz):
    s = hbarz / E_c
    return (s**2 + 4*s + 3 + 2 * cm.log(-s)) / (E_c * (1 + s)**3)

def fun_zero(z, Sigma):
    hbarz = hbar * z
    return special.wofz(np.sqrt(2 * sigma_sq)**-1 * (R_0(hbarz)**-1 - Sigma)) - 1j * np.sqrt(2 / np.pi) * np.sqrt(sigma_sq) * R_0(hbarz - Sigma)

def Sigma_0(z):
    return mp.findroot(lambda sig: fun_zero(z, sig), x0 = 0)
    # return optimize.root_scalar(lambda sig: fun_zero(z, sig), method = 'secant', x0 = 0, x1 = 0.)#.root

def mp_faddeeva(z):
    return mp.exp

print(Sigma_0(0.1))
    
# optimize.root_scalar(lambda x: x**2 - 1, method = 'secant', x0 = 0.5, x1 = 0.25)