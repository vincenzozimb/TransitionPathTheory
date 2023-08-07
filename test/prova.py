# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# import packages
from tpt.committor import potential

# define the grid
x = np.linspace(-1.5, 1.5, 100)  # Range of x values
y = np.linspace(-1, 1, 100)  # Range of y values
X, Y = np.meshgrid(x, y)
V = potential(X, Y)  # Compute the function values

# calculate the Gibbs distribution (beta=1)
y_low, y_high = (-1, 1)
x_low, x_high = (-1.5, 1.5)
Z, dZ = dblquad(potential, y_low, y_high, lambda x: x_low, lambda x: x_high)
prob = np.exp(-V) / Z

# plot the figure
plt.figure(figsize=(8, 6))
contourf = plt.contourf(X, Y, prob, levels=25, cmap='viridis')
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of Gibbs distribution of V(x, y)')
plt.show()