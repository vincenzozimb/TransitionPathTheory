# import libraries
import numpy as np
from scipy.integrate import dblquad
import csv
import pickle
import os

from functions import func 
from doublewell import dw

# import parameters
from main import beta, dx, Niter

# simulation box parameters
Lx, Ly = 1.5, 1
dy = dx
Nx = int(2*Lx/dx)
Ny = int(2*Ly/dy)

# create the grid
x = np.linspace(-Lx, Lx, Nx) # Range of x values
y = np.linspace(-Ly, Ly, Ny) # Range of y values
Y, X = np.meshgrid(y, x)  # Transpose X and Y here
V = dw.potential(X, Y) # potential
Vx = dw.V_partialx(X,Y) # derivative wrt x
Vy = dw.V_partialy(X,Y) # derivative wrt y

# calculate the Gibbs distribution
Z, dZ = dblquad(lambda y, x: np.exp(-beta*dw.potential(x, y)), -Ly, Ly, lambda x: -Lx, lambda x: Lx)
prob = np.exp(-beta*V) / Z

# metatable states
Vmax = 0.4

# initialize the committor
q = np.zeros((Nx,Ny))
func.neumann(q)
metastable_boolean = V < Vmax
reactant = x < 0
func.dirichlet(q, metastable_boolean, reactant)

# solve the BK equation
func.bk_solver(q, Vx, Vy, Nx, Ny, dx, dy, Niter, metastable_boolean, reactant)

# kernel of the probability density of reactive trajectories
m = q * (1-q) * np.exp(-V)

# transition path current
J = np.gradient(q) * prob

# reaction rate
dq_dx = np.gradient(q, axis=1)
integrand = np.exp(-beta * V[:, 0]) * dq_dx[:, 0]

k = np.sum(integrand) * dy / (Z * beta)
print("The reaction rate is k =", k)

# save data
parameters = {
    'Nx' : Nx,
    'Ny' : Ny,
    'Vmax' : Vmax
}

if not os.path.exists('doublewell/data'):
   os.makedirs('doublewell/data')

file_path = 'doublewell/data/parameters.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(parameters, file)

with open('doublewell/data/data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['X', 'Y', 'Potential', 'Gibbs', 'q', 'TPD', 'TPCx', 'TPCy'])  # Write header
    for x_row, y_row, V_row, P_row, q_row, m_row, Jx_row, Jy_row in zip(X, Y, V, prob, q, m, J[0], J[1]):
        for x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val in zip(x_row, y_row, V_row, P_row, q_row, m_row, Jx_row, Jy_row):
            csv_writer.writerow([x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val])
