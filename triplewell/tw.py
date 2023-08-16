# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# import variables
from main import beta, Niter, h

# define the potential
def potential(x):
    # kernel parameters
    x1 = -0.1
    x2 = 0.5
    x3 = 1.2
    x4 = 1.5
    x5 = -0.5
    x6 = 0.8
    # kernel
    def y(x):
        return (x - x1) * (x - x2) * (x - x3) * (x -x4) * (x - x5) * (x - x6)
    # shape parameters
    a = 10
    b = 1.5
    c = 0.1
    x0 = 0.552687
    # potential
    return a * y(b * x + x0) +c


# Generate x values and calculate the potential and the gibbs distribution
dx = h
L = 1
N = int(L / dx)
x = np.linspace(-L, L, N)
V = potential(x)
prob = np.exp(-beta*V)
Z = np.trapz(prob, x)
prob = prob / Z

print("(V0, V_minus, V_plus)", potential(0), potential(-1), potential(1))


# indentify reactant and product states
initial_guesses = [-2, 0, 0.5]
local_minima = []

for x0 in initial_guesses:
    result = minimize(potential, x0)
    local_minima.append((result.fun, result.x))

for i, (value, location) in enumerate(local_minima, start=1):
    print(f"Local Minimum {i}:")
    print("  Value:", value)
    print("  Location:", location)

V_basin, x_basin = 0.15, 0.35
R_bool = (np.abs(V-local_minima[1][0]) < V_basin) & (np.abs(x-local_minima[1][1]) < x_basin) 
xr = x[R_bool]

V_basin, x_basin = 0.6, 2
P1_bool = (np.abs(V-local_minima[0][0]) < V_basin) & (np.abs(x-local_minima[0][1]) < x_basin)
xp1 = x[P1_bool]

V_basin, x_basin = 0.5, 0.1
P2_bool = (np.abs(V-local_minima[2][0]) < V_basin) & (np.abs(x-local_minima[2][1]) < x_basin)
xp2 = x[P2_bool]

P_bool = P1_bool | P2_bool
metastable_bool = R_bool | P_bool 


# create images folder, if it does not already exist
if not os.path.exists('triplewell/images'):
   os.makedirs('triplewell/images')

# Plot of the potential and the Gibbs distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, V, label="Potential", color="blue")
plt.plot(xr, V[R_bool], label='Reactant', color="red")
plt.plot(xp1, V[P1_bool], label='Product 1', color="yellow")
plt.plot(xp2, V[P2_bool], label='Product 2', color="green")
plt.title('Asymmetric triple-well potential')
plt.xlim(-1, 1)
plt.ylim(-1, 1) 
plt.xlabel('x')
plt.ylabel('V(x)')
plt.legend()
plt.grid(True)

floor = np.zeros(N)
plt.subplot(1, 2, 2)
plt.plot(x, prob, label='pdf')
plt.plot(x,floor,color="gray")
plt.plot(xr, floor[R_bool], label='Reactant', color="red")
plt.plot(xp1, floor[P1_bool], label='Product 1', color="yellow")
plt.plot(xp2, floor[P2_bool], label='Product 2', color="green")
plt.title('Gibbs distribution')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("triplewell/images/p&g.png")


# Neumann boundary conditions on the edges of the simulation box
def neumann(q):
    """
    Apply Neumann boundary conditions on the edge of the simulation box: null derivative on the edge
    """
    # Dirichlet also
    q[0] = 1
    q[-1] = 1
    # Neumann boundary conditions on the edge
    q[1] = q[0]  # Forward finite difference for Neumann condition
    q[-2] = q[-1]  # Backward finite difference for Neumann condition


# Dirichlet boundary conditions on the metastable states
def dirichlet(q):
    """
    Apply Dirichlet boundary conditions to the committor function: 0 on reactant state and 1 on product state
    """
    # Apply Dirichlet boundary conditions on the metastable states
    q[R_bool] = 0
    q[P_bool] = 1


# iterative solver of 1D the backward Kolmogorov equation
eps = 1e-4  # Small step for numerical differentiation
def dVdx(x):
    return (potential(x+eps) - potential(x)) / eps

Vx = dVdx(x)

def bk_solver1D(q):
    """
    Solve iteratively the two-dimensional backward Kolmogorov using finite differences method  
    """
    for k in range(Niter):
        print(Niter-k, end="\r")
        for i in range(1, N - 1):
            if not metastable_bool[i]:
                q[i] = (q[i+1] + q[i-1]) / 2 - (q[i+1] - q[i-1]) * beta* Vx[i]* dx / 4
        # Neumann boundary conditions on the edge (x-axis)
        neumann(q)
        
        # Apply Dirichlet boundary conditions on the metastable states
        dirichlet(q)

q = np.zeros(N)
neumann(q)
dirichlet(q)
bk_solver1D(q)

# plots
m = prob * q * (1 - q)
J = np.gradient(q) * prob

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, q, label="Committor", color="blue")
plt.plot(x,floor,color="gray")
plt.plot(xr, floor[R_bool], label='Reactant', color="red")
plt.plot(xp1, floor[P1_bool], label='Product 1', color="yellow")
plt.plot(xp2, floor[P2_bool], label='Product 2', color="green")
plt.title('Committor function')
# plt.xlim(-1, 1)
# plt.ylim(-1, 1) 
plt.xlabel('x')
plt.ylabel('q(x)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, m, label='pdf')
plt.plot(x,floor,color="gray")
plt.plot(xr, floor[R_bool], label='Reactant', color="red")
plt.plot(xp1, floor[P1_bool], label='Product 1', color="yellow")
plt.plot(xp2, floor[P2_bool], label='Product 2', color="green")
plt.title('Transition path density')
plt.xlabel('x')
plt.ylabel('m(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("triplewell/images/q&m.png")


# current plot

# Subsample the data for arrow density
v = J
subsample_factor = 2  # Increase or decrease this value to control density
x_subsampled = x[::subsample_factor]
v_subsampled = v[::subsample_factor]

# Create a colormap based on the magnitude of the vectors
cmap = plt.get_cmap('viridis')

# Normalize the vector magnitudes to fit the colormap
norm = plt.Normalize(v.min(), v.max())

# Create a figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Line plot
ax.plot(x, v, color='b', label='Line Plot')

# Quiver plot (shows vector direction) with color mapping
quiver = ax.quiver(x_subsampled, np.zeros_like(x_subsampled), v_subsampled, np.zeros_like(v_subsampled), v_subsampled, angles='xy', scale_units='xy', scale=1, cmap=cmap, norm=norm, label='Quiver Plot')

# Add a colorbar to the quiver plot
cbar = plt.colorbar(quiver)
cbar.set_label('Vector Magnitude')

# Add a legend
ax.legend()

# Set plot title and labels
plt.title("Superimposed Line and Quiver Plots with Modified Arrow Density")
plt.xlabel("Position (x)")
plt.ylabel("Vector Magnitude (v)")
plt.grid()
plt.show()
