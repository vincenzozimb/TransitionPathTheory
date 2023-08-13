# import necessary packages
import numpy as np


# define the potential
def potential(x, y):
    return 5/2 * (x**2 - 1)**2 + 5*y**2


# gradient of the potential
def V_partialx(x, y):
    return 10*x*(x**2-1)

def V_partialy(x, y):
    return 10*y


# Neumann boundary conditions on the edges of the simulation box
def neumann(q):
    
    # Neumann boundary conditions on the edge (x-axis)
    q[0, :] = q[1, :]  # Forward finite difference for Neumann condition
    q[-1, :] = q[-2, :]  # Backward finite difference for Neumann condition
    
    # Neumann boundary conditions on the edge (y-axis)
    q[:, 0] = q[:, 1]  # Forward finite difference for Neumann condition
    q[:, -1] = q[:, -2]  # Backward finite difference for Neumann condition


# Dirichlet boundary conditions on the metastable states
def dirichlet(q, metastable_boolean, reactant):

    # Apply Dirichlet boundary conditions on the metastable states
    q[np.logical_and(metastable_boolean, reactant[:, np.newaxis])] = 0
    q[np.logical_and(metastable_boolean, ~reactant[:, np.newaxis])] = 1


# iterative solver of 2D the Backward-Kolmogorov equation
def bk_solver(q, Vx, Vy, Nx, Ny, dx, dy, Niter, metastable_boolean, reactant):
    
    for k in range(Niter):
        print(Niter-k, end="\r")
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if not metastable_boolean[i, j]:
                    q[i, j] = (2 * dx**2 * (q[i+1, j] + q[i-1, j]) + 2 * dy**2 * (q[i, j+1] + q[i, j-1])
                            - dx * dy**2 * Vx[i, j] * (q[i+1, j] - q[i-1, j])
                            - dy * dx**2 * Vy[i, j] * (q[i, j+1] - q[i, j-1])) / (4 * (dx**2 + dy**2))    
        # Neumann boundary conditions on the edge (x-axis)
        neumann(q)
        
        # Apply Dirichlet boundary conditions on the metastable states
        dirichlet(q, metastable_boolean, reactant)

