# import libraries
import numpy as np
import matplotlib.pyplot as plt
import os

from tw2d.tw_init import X, Y, V, prob, beta, R_bol, P_bol


# Create folder for images, if it does not already exist
if not os.path.exists('tw2d/images'):
   os.makedirs('tw2d/images')


# plot the potential and the Gibbs distribution
T_bol = R_bol | P_bol
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, V, levels=15, cmap='viridis')
plt.colorbar(label='Potential')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(T_bol, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=0.5)
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
plt.savefig("tw2d/images/p&g.png")


# import committor
q = np.load("tw2d/data.npy")



# kernel of the probability density of reactive trajectories
m = q * (1-q) * np.exp(-beta*V)
Zm = np.sum(m)
m = m / Zm


# plot q and m
# T_bol = R_bol | P_bol

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
contourf = plt.contourf(X, Y, q, levels=10, cmap='viridis')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(T_bol, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=0.2)
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Forward committor')

plt.subplot(1, 2, 2)
contourf = plt.contourf(X, Y, m, levels=10, cmap='viridis')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(T_bol, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=0.2)
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel of the transition path density')

plt.tight_layout()
plt.savefig("tw2d/images/q&m.png")

# transition path current
J = np.gradient(q) * prob / beta