import numpy as np
import mpmath as mp
import pylab as plt

hbar = 1

sigma_sq = 1
E_c = 1

def R_0(hbarz):
    s = hbarz / E_c
    return (s**2 + 4*s + 3 + 2 * mp.log(-s)) / (E_c * (1 + s)**3)

def self_consistency_Sigma(z, Sigma_Re, Sigma_Im):
    hbarz = hbar * z
    return mp_faddeeva(mp.sqrt(2 * sigma_sq)**-1 * (R_0(hbarz - (Sigma_Re + 1j * Sigma_Im))**-1 - (Sigma_Re + 1j * Sigma_Im))) - 1j * mp.sqrt(2 / np.pi) * mp.sqrt(sigma_sq) * R_0(hbarz - (Sigma_Re + 1j * Sigma_Im))

def Sigma_0(z):
    def f(sig_r, sig_i):
        return self_consistency_Sigma(z, sig_r, sig_i)
    return mp.findroot(f, (0,0), verify = False)

def mp_faddeeva(z):
    return mp.exp(-z**2)*mp.erfc(-1j * z)

length = 50
begin_z = -10
end_z = 10
zs = np.linspace(begin_z, end_z, length)

Real_Sigma = np.empty(length)
Imag_Sigma = np.empty(length)
for x in np.linspace(begin_z, end_z, length):
    print(x)
    ii = np.where(zs == x)
    evaluated = Sigma_0(x)
    Real_Sigma[ii] = np.real(evaluated)
    Imag_Sigma[ii] = np.imag(evaluated)
    
plt.ylim(-1,1)
plt.plot(zs,Real_Sigma)
plt.show()
plt.plot(zs,Imag_Sigma)
plt.show()

# optimize.root_scalar(lambda x: x**2 - 1, method = 'secant', x0 = 0.5, x1 = 0.25)