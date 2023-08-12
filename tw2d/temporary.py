# import libraries
import numpy as np
import matplotlib.pyplot as plt
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
Lx, Ly = 2, 1.5
dx = 0.06
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

# Visualize the potential and the Giggs distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, V, levels=20, cmap='viridis')
plt.colorbar(label='Potential')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potential Energy')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, prob, levels=20, cmap='viridis')
plt.colorbar(label='Probability density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gibbs distribution')

plt.tight_layout()
plt.savefig("tw2d/images/p&g.png")

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

# plot the committor
fig = plt.subplots(figsize=(8, 6))
contourf = plt.contourf(X, Y, q, levels=15, cmap='viridis')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(metastable_boolean, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=1)
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Forward committor')
plt.savefig("tw2d/images/c_tpt.png")


# kernel of the probability density of reactive trajectories
m = q * (1-q) * np.exp(-V)

# plot m
fig, ax = plt.subplots(figsize=(8, 6))
contourf = ax.contourf(X, Y, m, levels=15, cmap='viridis')
highlighted_contour = np.where(metastable_boolean, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel of the transition path density')
plt.savefig("tw2d/images/m_tpt.png")

# transition path current
J = np.gradient(q) * prob

# plot of the vector field
magnitude = np.sqrt(J[0]**2 + J[1]**2)
magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
plt.figure(figsize=(8, 6))
plt.streamplot(X.T, Y.T, J[0].T, J[1].T, color=magnitude_normalized.T, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.contourf(X, Y, highlighted_contour, colors='lightgray', alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transition Path Current')
# plt.gca().set_facecolor('lightgray')  # Set background color
plt.tight_layout()
plt.savefig("tw2d/images/J_tpt.png")













plt.show()