import numpy as np
from scipy import integrate
import pylab as plt

sigma_sq = 1
xi = 1
V_0 = 0

cutoff = np.inf

def W(r):
    return np.sqrt(2/np.pi) * (np.sqrt(sigma_sq) / xi) * np.exp(-r / xi)

def g(r):
    return (1 / sigma_sq) * integrate.quad(lambda r_p: W(r + r_p) * W(r_p), 0, cutoff)[0]

def sigma_1_sq(r):
    return sigma_sq * (1 - g(r)**2)

def Prob_Distr(V_1, r):
    return (1 / (np.sqrt(2 * np.pi * sigma_1_sq(r)))) * np.exp(- (V_1 - V_0 * g(r))**2 / (2 * sigma_1_sq(r)))

x = np.linspace(0,100,100)
vg = np.vectorize(Prob_Distr)
plt.plot(x,vg(x))

