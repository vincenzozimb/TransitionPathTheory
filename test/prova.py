import numpy as np
import matplotlib.pyplot as plt

# Create a grid of points
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Define the components of the vector field
def u(x, y):
    return -y

def v(x, y):
    return x

# Compute the magnitudes of the vector field
magnitude = np.sqrt(u(X, Y)**2 + v(X, Y)**2)

# Normalize the magnitudes for colormap mapping
magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

# Plot the streamlines with colored intensity
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, u(X, Y), v(X, Y), color=magnitude_normalized, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Streamlines with Colored Intensity')
plt.tight_layout()
plt.show()
