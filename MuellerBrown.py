import numpy as np
import matplotlib.pyplot as plt

# Known constants
A = np.array([-200, -100, -170, 15])
a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 0.6])
c = np.array([-10, -10, -6.5, 0.7])
x0 = np.array([1, 0, -0.5, -1])
y0 = np.array([0, 0.5, 1.5, 1])

# Recreate the mapping of the PES
xx = np.linspace(-1.8, 1.2, 400)
yy = np.linspace(-0.5, 2.2, 400)
X, Y = np.meshgrid(xx, yy)

# Mueller-Brown Potential Equation
W = (
    A[0] * np.exp(a[0] * ((X - x0[0]) ** 2) + b[0] * (X - x0[0]) * (Y - y0[0]) + c[0] * ((Y - y0[0]) ** 2)) +
    A[1] * np.exp(a[1] * ((X - x0[1]) ** 2) + b[1] * (X - x0[1]) * (Y - y0[1]) + c[1] * ((Y - y0[1]) ** 2)) +
    A[2] * np.exp(a[2] * ((X - x0[2]) ** 2) + b[2] * (X - x0[2]) * (Y - y0[2]) + c[2] * ((Y - y0[2]) ** 2)) +
    A[3] * np.exp(a[3] * ((X - x0[3]) ** 2) + b[3] * (X - x0[3]) * (Y - y0[3]) + c[3] * ((Y - y0[3]) ** 2))
)

Z = W - np.min(W)
Z[Z >= 200] = np.nan

# Plotting
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=100, cmap='magma')
plt.colorbar(contour, label='Energy')

# Add contour lines for visual clarity
contour_lines = plt.contour(X, Y, Z, colors='black', levels=np.arange(0, 180, 20))
plt.clabel(contour_lines, inline=1, fontsize=10, fmt='%d')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mueller-Brown Potential')
plt.tight_layout()
plt.show()

