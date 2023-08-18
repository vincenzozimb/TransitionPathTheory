# import necessary packages
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt


# define the potential and its gradient
def potential(x, y):
    # define the potential parameters
    A = 200
    a = 1.0
    x0 = [-1.5, 1.5]
    y0 = [0.0, 0.0]
    k = 20.0
    # calculate expression
    V = (
        A * np.exp(-a * ((x - x0[0])**2 + (y - y0[0])**2)) -
        A * np.exp(-a * ((x - x0[1])**2 + (y - y0[1])**2))
    )
    V += k * (x**2) * (y**2)
    # return
    return V

def V_partialx(x, y):
    eps = 1e-4
    return (potential(x+eps, y) - potential(x-eps, y)) / (2 * eps)

def V_partialy(x, y):
    eps = 1e-4
    return (potential(x, y+eps) - potential(x, y-eps)) / (2 * eps)


# simulation parameters
Lx = (-4.0, 4.0)
Ly = (-3.0, 3.0)
dx = 0.05
dy = dx

Nx = int((Lx[1] - Lx[0]) / dx)
Ny = int((Ly[1] - Ly[0]) / dy)

beta = 0.005 
VmaxR = 80.0
VmaxP = -100.0
Niter = 1000


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
R_bol = (V < VmaxR) & (X < -2.0)
P_bol = (V < VmaxP) & (X > 0.0)
