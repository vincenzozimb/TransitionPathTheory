# import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# simulation box parameters
a = 1 
L = 0.8
h = 0.05
Nx = int(L/h)
Ny = int(a/h)

# create the grid
x = np.linspace(0, L, Nx) # Range of x values
y = np.linspace(0, a, Ny) # Range of y values
Y, X = np.meshgrid(y, x)  # Transpose X and Y here

# initialize the potential
v = np.zeros((Nx, Ny))

# impose boundary conditions
v[0, :] = 1
v[-1, :] = 0
v[:, 0] = 0
v[:, -1] = 0

# update with iterations
Niter = 500
for k in range(Niter):
    print(k)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if i == 0:
                v[i, j] = (v[i+1, j] + v[i, j+1] + v[i, j-1]) / 3
            elif i == Nx - 1:
                v[i, j] = (v[i-1, j] + v[i, j+1] + v[i, j-1]) / 3
            elif j == 0:
                v[i, j] = (v[i+1, j] + v[i-1, j] + v[i, j+1]) / 3
            elif j == Ny - 1:
                v[i, j] = (v[i+1, j] + v[i-1, j] + v[i, j-1]) / 3
            else:
                v[i, j] = (v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1]) / 4

# Create a figure with subplots
fig = plt.figure(figsize=(12, 6))

# Plot the 3D surface plot in the first subplot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, v, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Potential')
ax1.set_title('3D Surface Plot')

# Plot the contour plot in the second subplot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, v, levels=10, cmap='viridis')
fig.colorbar(contour, ax=ax2, label='Potential')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()