from inverse import inv_run
from inverse import inv_analyse


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define the potential parameters
# A_left = 100  # Lower value for the left well
# A_right = 100
# a = 1.0
# x0_left = [-0.5, 1.5]  # Shifted to the left
# x0_right = [1.5, 0.5]
# y0 = [0.0, 0.0]
# k = 20.0

# # Create a meshgrid for x and y coordinates
# x = np.linspace(-3, 3, 400)
# y = np.linspace(-2, 2, 400)
# X, Y = np.meshgrid(x, y)

# # Define the potential energy surface
# V = (
#     A_left * np.exp(-a * ((X - x0_left[0])**2 + (Y - y0[0])**2)) -
#     A_right * np.exp(-a * ((X - x0_right[0])**2 + (Y - y0[0])**2))
# )

# # # Define the potential parameters
# # A = 100
# # a = 1.0
# # x0 = [-1.5, 1.5]
# # y0 = [0.0, 0.0]
# # k = 20.0

# # # Create a meshgrid for x and y coordinates
# # x = np.linspace(-4, 4, 400)
# # y = np.linspace(-3, 3, 400)
# # X, Y = np.meshgrid(x, y)

# # # Define the potential energy surface
# # V = (
# #     A * np.exp(-a * ((X - x0[0])**2 + (Y - y0[0])**2)) -
# #     A * np.exp(-a * ((X - x0[1])**2 + (Y - y0[1])**2))
# # )

# # Create a barrier between the wells
# V += k * (X**2) * (Y**2)
# V[V > 200] = np.nan

# # Plot the surface plot
# fig = plt.figure(figsize=(12, 6))

# # Surface plot
# ax1 = fig.add_subplot(121, projection='3d')
# surface = ax1.plot_surface(X, Y, V, cmap='coolwarm', rstride=5, cstride=5, alpha=0.8, antialiased=True)
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Potential Energy')
# ax1.set_title('3D Potential Energy Surface')
# fig.colorbar(surface, ax=ax1, shrink=0.6)

# # Contour plot
# ax2 = fig.add_subplot(122)
# contour = ax2.contourf(X, Y, V, levels=50, cmap='coolwarm')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_title('Contour Plot of Potential Energy')
# fig.colorbar(contour, ax=ax2, label='Potential Energy')

# plt.tight_layout()
# plt.show()

