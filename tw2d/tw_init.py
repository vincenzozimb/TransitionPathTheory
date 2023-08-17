# import necessary packages
import numpy as np
from scipy.integrate import dblquad


# define the potential and its gradient
def potential(x, y):
    x0 = 1
    a0 = 1 / 3
    b0 = 5 / 3
    u0 = 5
    w0 = 1 / 5
    return u0 * (np.exp(-(x**2 + y**2)) - 3 / 5 * np.exp(-(x**2 + (y - b0)**2)) - np.exp(-((x - x0)**2 + y**2)) - np.exp(-((x + x0)**2 + y**2)) ) + w0 * (x**4 + (y-a0)**4)

def V_partialx(x, y):
    eps = 1e-4
    return (potential(x+eps, y) - potential(x-eps, y)) / (2 * eps)

def V_partialy(x, y):
    eps = 1e-4
    return (potential(x, y+eps) - potential(x, y-eps)) / (2 * eps)

# def potential(x, y):
#     term1 = 3 * np.exp(-x**2 - (y - 1/3)**2)
#     term2 = 3 * np.exp(-x**2 - (y - 5/3)**2)
#     term3 = 5 * np.exp(-(x - 1)**2 - y**2)
#     term4 = 5 * np.exp(-(x + 1)**2 - y**2)
#     term5 = 0.2 * x**4
#     term6 = 0.2 * (y - 1/3)**4

#     return term1 - term2 - term3 - term4 + term5 + term6

# def V_partialx(x, y):
#     term1 = -6 * x * np.exp(-x**2 - (y - 1/3)**2)
#     term2 = -6 * x * np.exp(-x**2 - (y - 5/3)**2)
#     term3 = -10 * (x - 1) * np.exp(-(x - 1)**2 - y**2)
#     term4 = -10 * (x + 1) * np.exp(-(x + 1)**2 - y**2)
#     term5 = 0.8 * x**3
#     term6 = 0

#     return term1 - term2 - term3 - term4 + term5 + term6

# def V_partialy(x, y):
    term1 = -6 * (y-1/3) * np.exp(-x**2 - (y - 1/3)**2)
    term2 = -6 * (y-5/3) * np.exp(-x**2 - (y - 5/3)**2)
    term3 = -10 * y * np.exp(-(x - 1)**2 - y**2)
    term4 = -10 * y * np.exp(-(x + 1)**2 - y**2)
    term5 = 0
    term6 = 0.8 * (y - 1/3)**3

    return term1 - term2 - term3 - term4 + term5 + term6


# simulation parameters
N = 20
Lx = (-2.0, 2.0)
Ly = (-1.5, 2.5)
beta = 1 / 0.6
Vmax = -2.5
Niter = 10000


# create the grid
x = np.linspace(Lx[0], Lx[1], N)
y = np.linspace(Ly[0], Ly[1], N)

X, Y = np.meshgrid(x, y)

dx = (Lx[1]-Lx[0]) / N
dy = (Ly[1]-Ly[0]) / N

print(dx, dy)


# calculate the potential and its derivatives
V = potential(X, Y)
Vx = V_partialx(X, Y)
Vy = V_partialy(X, Y)

# calculate the Gibbs distribution
Z, dZ = dblquad(lambda y, x: np.exp(-beta*potential(x, y)), Ly[0], Ly[1], lambda x: Lx[0], lambda x: Lx[1])
prob = np.exp(-beta*V) / Z


# identify metastable states
R_bol = (V < Vmax) & (x < 0.0)
P_bol = (V < Vmax) & (x > 0.0)
