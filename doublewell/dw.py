# import necessary packages
import numpy as np
import pickle


# import parameters
Vmax = 0.4

# define the potential
def potential(x, y):
    return 5/2 * (x**2 - 1)**2 + 5 * y**2


# gradient of the potential
def V_partialx(x, y):
    return 10 * x * (x**2 - 1)

def V_partialy(x, y):
    return 10 * y


# boolean indicator functions of the metastable states
def in_reac(x, y):
    return x < 0 and potential(x, y) < Vmax  # condition for reactant set 

def in_prod(x, y):
    return x > 0 and potential(x, y) < Vmax  # condition for product set 


# first and last passage time functions (return the index)
def t_plus(t0, t, x, y):
    for n in range(len(t)):
        if t[n] < t0:
            continue
        if in_reac(x[n], y[n]) or in_prod(x[n], y[n]):
            return n
    return np.inf

def t_minus(t0, t, x, y):
    for n in reversed(range(len(t))):
        if t[n] > t0:
            continue
        if in_reac(x[n], y[n]) or in_prod(x[n], y[n]):
            return n
    return -np.inf
