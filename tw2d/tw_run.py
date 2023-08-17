# import libraries
import numpy as np

from tw2d.tw_init import Vx, Vy, dx, dy, R_bol, P_bol, Niter, Nx, Ny, beta


# Neumann boundary conditions on the edge of the simulation box (null normal derivative)
def neumann(q):
    # Neumann boundary conditions on the edge (x-axis)
    q[0, :] = q[1, :]  # Forward finite difference on the left
    q[-1, :] = q[-2, :]  # Backward finite difference on the right
    
    # Neumann boundary conditions on the edge (y-axis)
    q[:, 0] = q[:, 1]  # Forward finite difference on the bottom
    q[:, -1] = q[:, -2]  # Backward finite difference on the top


# Dirichlet boundary conditions on the metastable states
def dirichlet(q):
    q[R_bol] = 0
    q[P_bol] = 1


# initialize the committor
q = np.full((Nx, Ny), 0.5)
neumann(q)
dirichlet(q)


# update with iterations
# for k in range(Niter):
#     print(Niter-k, end="\r")
#     for i in range(1, N - 1):
#         for j in range(1, N - 1):
#             if not (R_bol[i, j] | P_bol[i, j]):
#                 q[i, j] = (2 * dx**2 * (q[i+1, j] + q[i-1, j]) + 2 * dy**2 * (q[i, j+1] + q[i, j-1])
#                         - dx * dy**2 * beta * Vx[i, j] * (q[i+1, j] - q[i-1, j])
#                         - dy * dx**2 * beta * Vy[i, j] * (q[i, j+1] - q[i, j-1])) / (4 * (dx**2 + dy**2)) 
#     # Apply boundary conditions
#     neumann(q)
#     dirichlet(q)


for k in range(Niter):
    print(Niter-k, end="\r")
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if not (R_bol[i, j] | P_bol[i, j]):
                q[i, j] = (dy**2 * (q[i+1, j] + q[i-1, j]) + dx**2 * (q[i, j+1] + q[i, j-1]) 
                        - beta * dx * dy * (dy * Vx[i ,j] * (q[i+1, j] - q[i-1, j]) + dx * Vy[i ,j] * (q[i, j+1] - q[i, j-1])) / 2 ) / (2 * (dx**2 + dy**2))    
    # Apply boundary conditions
    neumann(q)
    dirichlet(q)


# save data
np.save("tw2d/data.npy", q)
