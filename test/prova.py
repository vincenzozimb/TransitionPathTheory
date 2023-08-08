# import libraries
import numpy as np
import matplotlib.pyplot as plt

# simulation box parameters
Lx = 1 
Ly = 1
h = 0.05
Nx = int(Lx/h)
Ny = int(Ly/h)

# create the grid
x = np.linspace(0, Lx, Nx) # Range of x values
y = np.linspace(0, Ly, Ny) # Range of y values
X, Y = np.meshgrid(x, y)  # Transpose X and Y here

# initialize grid to zero
z = np.zeros((Nx, Ny))

for i in range(int(Nx/4)):
    for j in range(int(Ny/3)):
        z[i,j] = 1

# plot the figure
plt.figure(figsize=(8, 6))
contourf = plt.contourf(X, Y, z, levels=10, cmap='viridis')
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()