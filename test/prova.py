# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# define the potential
def potential(x, y):
    return 5/2 * (x**2 - 1)**2 + 5*y**2

# gradient of the potential
def V_x(x, y):
    return 10*x*(x**2-1)

def V_y(x, y):
    return 10*y

# simulation box parameters
Lx, Ly = 1.5, 1
dx = 0.05
dy = dx
Nx = int(2*Lx/dx)
Ny = int(2*Ly/dy)

# create the grid
x = np.linspace(-Lx, Lx, Nx) # Range of x values
y = np.linspace(-Ly, Ly, Ny) # Range of y values
Y, X = np.meshgrid(y, x)  # Transpose X and Y here
V = potential(X, Y) # potential
Vx = V_x(X,Y) # derivative wrt x
Vy = V_y(X,Y) # derivative wrt y

# metatable states
Vmax = 0.4

# initialize the committor
q = np.zeros((Nx,Ny))

# Neumann boundary conditions on the edge
q[0,:] = q[1,:]
q[-1,:] = q[-2,:]
q[:,0] = q[:,1]
q[:,-1] = q[:,-2]

# Dirichlet boundary conditions on the metastable states
metastable_boolean = V < Vmax
reactanct = x < 0
q[np.logical_and(metastable_boolean, reactanct[:, np.newaxis])] = 0
q[np.logical_and(metastable_boolean, ~reactanct[:, np.newaxis])] = 1

# update with iterations
Niter = 1000
for k in range(Niter):
    print(Niter-k, end="\r")
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if not metastable_boolean[i, j]:
                q[i, j] = (2 * dx**2 * (q[i+1, j] + q[i-1, j]) + 2 * dy**2 * (q[i, j+1] + q[i, j-1])
                           - dx * dy**2 * Vx[i, j] * (q[i+1, j] - q[i-1, j])
                           - dy * dx**2 * Vy[i, j] * (q[i, j+1] - q[i, j-1])) / (4 * (dx**2 + dy**2))    
    # Neumann boundary conditions on the edge (x-axis)
    q[0, :] = q[1, :]  # Forward finite difference for Neumann condition
    q[-1, :] = q[-2, :]  # Backward finite difference for Neumann condition
    
    # Neumann boundary conditions on the edge (y-axis)
    q[:, 0] = q[:, 1]  # Forward finite difference for Neumann condition
    q[:, -1] = q[:, -2]  # Backward finite difference for Neumann condition
    
    # Apply Dirichlet boundary conditions on the metastable states
    q[np.logical_and(metastable_boolean, reactanct[:, np.newaxis])] = 0
    q[np.logical_and(metastable_boolean, ~reactanct[:, np.newaxis])] = 1

# plot the committor
qplot = q
qplot[metastable_boolean] = 0
fig, ax = plt.subplots(figsize=(8, 6))
contourf = ax.contourf(X, Y, q, levels=15, cmap='viridis')
contour = ax.contour(X, Y, qplot, levels=[0.5], color="black")
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("fig.png")
plt.show()

# probability density of reactive trajectories
m = q * (1-q) * np.exp(-V)

# plot m
fig, ax = plt.subplots(figsize=(8, 6))
contourf = ax.contourf(X, Y, m, levels=15, cmap='viridis')
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()