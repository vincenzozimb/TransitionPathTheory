# import libraries
import numpy as np
from scipy.integrate import dblquad
import csv
import pickle

# define the potential
def potential(x, y):
    return 5/2 * (x**2 - 1)**2 + 5*y**2

# gradient of the potential
def V_partialx(x, y):
    return 10*x*(x**2-1)

def V_partialy(x, y):
    return 10*y

# simulation box parameters
Lx, Ly = 1.5, 1
dx = 0.03
dy = dx
Nx = int(2*Lx/dx)
Ny = int(2*Ly/dy)

# create the grid
x = np.linspace(-Lx, Lx, Nx) # Range of x values
y = np.linspace(-Ly, Ly, Ny) # Range of y values
Y, X = np.meshgrid(y, x)  # Transpose X and Y here
V = potential(X, Y) # potential
Vx = V_partialx(X,Y) # derivative wrt x
Vy = V_partialy(X,Y) # derivative wrt y

# calculate the Gibbs distribution (beta=1)
Z, dZ = dblquad(lambda y, x: np.exp(-potential(x, y)), -Ly, Ly, lambda x: -Lx, lambda x: Lx)
beta = 1
prob = np.exp(-beta*V) / Z

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
Niter = 3000
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

# kernel of the probability density of reactive trajectories
m = q * (1-q) * np.exp(-V)

# transition path current
J = np.gradient(q) * prob

# save data
parameters = {
    'Nx' : Nx,
    'Ny' : Ny,
    'Vmax' : Vmax
}

file_path = 'dw2d/parameters.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(parameters, file)

with open('dw2d/data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X', 'Y', 'Potential', 'Gibbs', 'q', 'TPD', 'TPCx', 'TPCy'])  # Write header
    for x_row, y_row, V_row, P_row, q_row, m_row, Jx_row, Jy_row in zip(X, Y, V, prob, q, m, J[0], J[1]):
        for x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val in zip(x_row, y_row, V_row, P_row, q_row, m_row, Jx_row, Jy_row):
            csv_writer.writerow([x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val])
