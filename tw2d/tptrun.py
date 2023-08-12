# import libraries
import numpy as np
from scipy.integrate import dblquad
import csv
import pickle

# define the potential
def potential(x, y):
    term1 = 3 * np.exp(-x**2 - (y - 1/3)**2)
    term2 = 3 * np.exp(-x**2 - (y - 5/3)**2)
    term3 = 5 * np.exp(-(x - 1)**2 - y**2)
    term4 = 5 * np.exp(-(x + 1)**2 - y**2)
    term5 = 0.2 * x**4
    term6 = 0.2 * (y - 1/3)**4

    return term1 - term2 - term3 - term4 + term5 + term6

# gradient of the potential
def V_partialx(x, y):
    term1 = -6 * x * np.exp(-x**2 - (y - 1/3)**2)
    term2 = -6 * x * np.exp(-x**2 - (y - 5/3)**2)
    term3 = -10 * (x - 1) * np.exp(-(x - 1)**2 - y**2)
    term4 = -10 * (x + 1) * np.exp(-(x + 1)**2 - y**2)
    term5 = 0.8 * x**3
    term6 = 0

    return term1 - term2 - term3 - term4 + term5 + term6

def V_partialy(x, y):
    term1 = -6 * (y-1/3) * np.exp(-x**2 - (y - 1/3)**2)
    term2 = -6 * (y-5/3) * np.exp(-x**2 - (y - 5/3)**2)
    term3 = -10 * y * np.exp(-(x - 1)**2 - y**2)
    term4 = -10 * y * np.exp(-(x + 1)**2 - y**2)
    term5 = 0
    term6 = 0.8 * (y - 1/3)**3

    return term1 - term2 - term3 - term4 + term5 + term6

# simulation box parameters
Lx, Ly = 2, 2.5
dx = 0.1
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
beta = 6.67
Z, dZ = dblquad(lambda y, x: np.exp(-beta*potential(x, y)), -Ly, Ly, lambda x: -Lx, lambda x: Lx)
prob = np.exp(-beta*V) / Z

# metatable states
Vmax = -3.2

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
Niter = 2000
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
m = q * (1-q) * np.exp(-beta*V)

# transition path current
J = np.gradient(q) * prob

# save data
parameters = {
    'Nx' : Nx,
    'Ny' : Ny,
    'Vmax' : Vmax
}

file_path = 'tw2d/parameters.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(parameters, file)

with open('tw2d/data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X', 'Y', 'Potential', 'Gibbs', 'q', 'TPD', 'TPCx', 'TPCy'])  # Write header
    for x_row, y_row, V_row, P_row, q_row, m_row, Jx_row, Jy_row in zip(X, Y, V, prob, q, m, J[0], J[1]):
        for x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val in zip(x_row, y_row, V_row, P_row, q_row, m_row, Jx_row, Jy_row):
            csv_writer.writerow([x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val])
