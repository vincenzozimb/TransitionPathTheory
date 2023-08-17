# import necessary packages
import numpy as np
from scipy.integrate import dblquad


# define the potential and its gradient
def potential(x, y):
    return 5/2 * (x**2 - 1)**2 + 5 * y**2

def V_partialx(x, y):
    return 10 * x * (x**2 - 1)

def V_partialy(x, y):
    return 10 * y

# simulation parameters
Lx = (-1.5, 1.5)
Ly = (-1.0, 1.0)
dx = 0.03
dy = dx

Nx = int((Lx[1] - Lx[0]) / dx)
Ny = int((Ly[1] - Ly[0]) / dy)

beta = 1 
Vmax = 0.4
Niter = 3000


# create the grid
x = np.linspace(Lx[0], Lx[1], Nx)
y = np.linspace(Ly[0], Ly[1], Ny)

Y, X = np.meshgrid(y, x) 


# calculate the potential and its derivatives
V = potential(X, Y)
Vx = V_partialx(X, Y)
Vy = V_partialy(X, Y)


# calculate the Gibbs distribution
Z, dZ = dblquad(lambda y, x: np.exp(-beta*potential(x, y)), Ly[0], Ly[1], lambda x: Lx[0], lambda x: Lx[1])
prob = np.exp(-beta*V) / Z


# identify metastable states
R_bol = (V < Vmax) & (X < 0.0)
P_bol = (V < Vmax) & (X > 0.0)
