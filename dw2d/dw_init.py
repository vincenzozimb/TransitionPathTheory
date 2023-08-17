# import necessary packages
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt


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


# Euler-Maruyama scheme for overdamped Langevin equation
def overdamped_langevin(x0, y0, fx, fy, D, dt, T):
    """
    Solves the 2D overdamped Langevin equation using the Euler-Maruyama method.

    Parameters:
        x0, y0: float
            Initial position of the particle in the x and y directions.
        fx, fy: function
            Functions that calculate the deterministic force as a function of position.
        D: float
            Diffusion coefficient representing the strength of the random force (assuming it's equal in both dimensions).
        dt: float
            Time step for the numerical integration.
        T: float
            Total time for which to integrate.

    Returns:
        t: numpy array
            Time points of the integration.
        x, y: numpy arrays
            Particle positions at each time point.
    """
    num_steps = int(T / dt)
    t = np.linspace(0.0, T, num_steps + 1)
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    x[0], y[0] = x0, y0

    for i in range(num_steps):
        wx, wy = np.random.normal(0.0, np.sqrt(dt), 2)
        x[i + 1] = x[i] - fx(x[i], y[i]) * dt + np.sqrt(2 * D) * wx
        y[i + 1] = y[i] - fy(x[i], y[i]) * dt + np.sqrt(2 * D) * wy

    return t, x, y



# Initial positions (x, y)
x0, y0 = -1.0, 0.0 


# Define deterministic forces for the x and y directions
f_x = lambda x, y: V_partialx(x, y)
f_y = lambda x, y: V_partialy(x, y)


# Solve the 2D Langevin equation
dt = 0.01
T = 10.0
D = 1 / beta
t, x, y = overdamped_langevin(x0, y0, f_x, f_y, D, dt, T)


# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, V, levels=15, cmap='coolwarm')
plt.colorbar(label='Energy')
plt.plot(x, y, linestyle='-', color='blue', label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Energy Landscape with Trajectory')
plt.legend()
plt.savefig("dw2d/images/traj.png")