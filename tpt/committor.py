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
N = 100
dx = 2*Lx/N
dy = 2*Ly/N

# create the grid
x = np.linspace(-Lx, Lx, N) # Range of x values
y = np.linspace(-Ly, Ly, N) # Range of y values
X, Y = np.meshgrid(x, y)
V = potential(X, Y) # potential
Vx = V_x(X,Y) # derivative wrt x
Vy = V_y(X,Y) # derivative wrt y

# calculate the Gibbs distribution (beta=1)
Z, dZ = dblquad(lambda y, x: np.exp(-potential(x, y)), -Ly, Ly, lambda x: -Lx, lambda x: Lx)
prob = np.exp(-V) / Z

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
plt.show()

# metatable states
Vmax = 0.4

# initialize the committor
q = np.zeros((N,N))

# Neumann boundary conditions on the edge
q[0,:] = q[1,:]
q[-1,:] = q[-2,:]
q[:,0] = q[:,1]
q[:,-1] = q[:,-2]

# Dirichlet boundary conditions on the metastable states
metastable_boolean = V < Vmax
q[metastable_boolean] = 0 # wrong

# update with iterations
Niter = 500
for k in range(Niter):
    print(k)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if not metastable_boolean[i, j]:
                q[i, j] = (2 * dx**2 * (q[i+1, j] + q[i-1, j]) + 2 * dy**2 * (q[i, j+1] + q[i, j-1])
                           - dx * dy**2 * Vx[i, j] * (q[i+1, j] - q[i-1, j])
                           - dy * dx**2 * Vy[i, j] * (q[i, j+1] - q[i, j-1])) / (4 * (dx**2 + dy**2))
            else:
                q[i,j] = 0
    
    # Neumann boundary conditions on the edge (x-axis)
    q[0, :] = q[1, :] - dx * Vy[0, :]  # Forward finite difference for Neumann condition
    q[-1, :] = q[-2, :] + dx * Vy[-1, :]  # Backward finite difference for Neumann condition
    
    # Neumann boundary conditions on the edge (y-axis)
    q[:, 0] = q[:, 1] - dy * Vx[:, 0]  # Forward finite difference for Neumann condition
    q[:, -1] = q[:, -2] + dy * Vx[:, -1]  # Backward finite difference for Neumann condition

# plot the committor
plt.figure(figsize=(8, 6))
contourf = plt.contourf(X, Y, q, levels=10, cmap='viridis')
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()