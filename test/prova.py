import numpy as np
import matplotlib.pyplot as plt

# Example data for the vector field (1D)
x = np.linspace(0, 10, 100)  # High-resolution positions along the x-axis
v = np.sin(x)  # Magnitude of the vector at each position

# Subsample the data for arrow density
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




# cmap = plt.get_cmap('viridis')  # You can choose any colormap you prefer
# norm = plt.Normalize(J.min(), J.max())
# plt.figure(figsize=(8, 6))
# ax = plt.gca()
# quiv = ax.quiver(x, np.zeros_like(x), J, np.zeros_like(x), J, angles='xy', scale_units='xy', scale=1, cmap=cmap, norm=norm)
# cbar = plt.colorbar(quiv)
# cbar.set_label('Vector Magnitude')

# plt.title("1D Vector Field with Color Mapped Intensity")
# plt.xlabel("Position (x)")
# plt.ylabel("Vector Magnitude (v)")
# plt.grid()
# plt.show()
