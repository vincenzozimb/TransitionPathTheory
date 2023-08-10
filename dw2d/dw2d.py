# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

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
plt.savefig("dw2d/p&g.png")

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
plt.savefig("dw2d/c_tpt.png")


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
plt.savefig("dw2d/m_tpt.png")

# transition path current
J = np.gradient(q) * prob

# plot of the vector field
subsample_factor = 2
X_plot = X[::subsample_factor, ::subsample_factor]
Y_plot = Y[::subsample_factor, ::subsample_factor]
Jx_plot = J[0][::subsample_factor, ::subsample_factor]
Jy_plot = J[1][::subsample_factor, ::subsample_factor]

plt.figure(figsize=(8, 6))
plt.quiver(X_plot, Y_plot, Jx_plot, Jy_plot, angles='xy', scale_units='xy', scale=0.1, color='darkblue', alpha=0.7, width=0.005, headwidth=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transition path current')
plt.gca().set_facecolor('lightgray')  # Set background color
plt.savefig("dw2d/J_tpt.png")
