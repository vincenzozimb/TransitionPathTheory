import numpy as np
import matplotlib.pyplot as plt

# Generate example data
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Define the components of the vector field (u, v)
u = -Y
v = X

# Create a quiver plot with improved styling
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v, angles='xy', scale_units='xy', scale=1.5, color='darkblue', alpha=0.7, headwidth=4)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Vector Field')
# plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_facecolor('lightgray')  # Set background color
# plt.axhline(0, color='black', linewidth=0.8)
# plt.axvline(0, color='black', linewidth=0.8)

plt.show()
