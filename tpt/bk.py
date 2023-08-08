import numpy as np

# Define grid parameters
Nx = 50
Ny = 50
Lx = 1.0
Ly = 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# create the grid
x = np.linspace(-Lx, Lx, Nx) # Range of x values
y = np.linspace(-Ly, Ly, Ny) # Range of y values
X, Y = np.meshgrid(x, y)

# Initialize solution q
q = np.zeros((Ny, Nx))

# gradient of the potential
def V_x(x, y):
    return 10*x*(x**2-1)

def V_y(x, y):
    return 10*y

# Perform iterative updates
num_iterations = 1000
for _ in range(num_iterations):
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            q_x = (q[i, j+1] - q[i, j-1]) / (2 * dx)
            q_y = (q[i+1, j] - q[i-1, j]) / (2 * dy)
            q_xx = (q[i, j+1] - 2 * q[i, j] + q[i, j-1]) / dx**2
            q_yy = (q[i+1, j] - 2 * q[i, j] + q[i-1, j]) / dy**2
            q[i, j] = (V_x(x[j], y[i]) * q_x + V_y(x[j], y[i]) * q_y - q_xx - q_yy) * dx**2 + q[i, j]

# Visualize the result using Matplotlib
import matplotlib.pyplot as plt
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, q, levels=20, cmap='viridis')
plt.colorbar(label='q')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of PDE')
plt.show()
