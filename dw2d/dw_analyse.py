# import libraries
import numpy as np
import matplotlib.pyplot as plt
import os

from dw2d.dw_init import X, Y, V, prob, beta, R_bol, P_bol


# Create folder for images, if it does not already exist
if not os.path.exists('dw2d/images'):
   os.makedirs('dw2d/images')


# useful variables
T_bol = R_bol | P_bol


# plot the potential and the Gibbs distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, V, levels=15, cmap='viridis')
plt.colorbar(label='Potential')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(T_bol, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=0.0)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potential Energy')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, prob, levels=25, cmap='viridis')
plt.colorbar(label='Probability density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gibbs distribution')

plt.tight_layout()
plt.savefig("dw2d/images/p&g.png")


# import committor
q = np.load("dw2d/data.npy")


# kernel of the probability density of reactive trajectories
m = q * (1-q) * np.exp(-beta*V)
# Zm = np.sum(m)
# m = m / Zm


# plot q and m
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
contourf = plt.contourf(X, Y, q, levels=10, cmap='viridis')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(T_bol, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=0.5)
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Forward committor')

plt.subplot(1, 2, 2)
contourf = plt.contourf(X, Y, m, levels=10, cmap='viridis')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(T_bol, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=0.5)
# plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel of the transition path density')

plt.tight_layout()
plt.savefig("dw2d/images/q&m.png")

# transition path current
J = np.gradient(q) * prob / beta

magnitude = np.sqrt(J[0]**2 + J[1]**2)
magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
plt.figure(figsize=(8, 6))
plt.streamplot(X.T, Y.T, J[0].T, J[1].T, density=1.2, color=magnitude_normalized.T, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.contourf(X, Y, highlighted_contour, colors='lightgray', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transition Path Current')
plt.gca().set_facecolor('snow')  # Set background color
plt.tight_layout()
plt.savefig("dw2d/images/J_tpt.png")