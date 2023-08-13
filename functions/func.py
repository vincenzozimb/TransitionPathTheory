# import necessary packages
import numpy as np


# Neumann boundary conditions on the edges of the simulation box
def neumann(q):
    """
    Apply Neumann boundary conditions on the edge of the simulation box: null-normal derivative on the edge
    """
    # Neumann boundary conditions on the edge (x-axis)
    q[0, :] = q[1, :]  # Forward finite difference for Neumann condition
    q[-1, :] = q[-2, :]  # Backward finite difference for Neumann condition
    
    # Neumann boundary conditions on the edge (y-axis)
    q[:, 0] = q[:, 1]  # Forward finite difference for Neumann condition
    q[:, -1] = q[:, -2]  # Backward finite difference for Neumann condition


# Dirichlet boundary conditions on the metastable states
def dirichlet(q, metastable_boolean, reactant):
    """
    Apply Dirichlet boundary conditions to the committor function: 0 on reactant state and 1 on product state
    """
    # Apply Dirichlet boundary conditions on the metastable states
    q[np.logical_and(metastable_boolean, reactant[:, np.newaxis])] = 0
    q[np.logical_and(metastable_boolean, ~reactant[:, np.newaxis])] = 1


# iterative solver of 2D the backward Kolmogorov equation
def bk_solver(q, Vx, Vy, Nx, Ny, dx, dy, Niter, metastable_boolean, reactant):
    """
    Solve iteratively the two-dimensional backward Kolmogorov using finite differences method  
    """
    for k in range(Niter):
        print(Niter-k, end="\r")
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if not metastable_boolean[i, j]:
                    q[i, j] = (2 * dx**2 * (q[i+1, j] + q[i-1, j]) + 2 * dy**2 * (q[i, j+1] + q[i, j-1])
                            - dx * dy**2 * Vx[i, j] * (q[i+1, j] - q[i-1, j])
                            - dy * dx**2 * Vy[i, j] * (q[i, j+1] - q[i, j-1])) / (4 * (dx**2 + dy**2))    
        # Neumann boundary conditions on the edge (x-axis)
        neumann(q)
        
        # Apply Dirichlet boundary conditions on the metastable states
        dirichlet(q, metastable_boolean, reactant)


def overdamped_langevin(x0, y0, fx, fy, D, dt, T):
    """
    Solves the 2D overdamped Langevin equation using the Euler-Maruyama method.

    Parameters:
        x0, y0: float
            Initial position of the particle in the x and y directions.
        fx, fy: function
            Functions that calculate the deterministic force as a function of position.
        D: float
            Diffusion coefficient representing the strength of the random force (assuming it's equal in both dimensions).
        dt: float
            Time step for the numerical integration.
        T: float
            Total time for which to integrate.

    Returns:
        t: numpy array
            Time points of the integration.
        x, y: numpy arrays
            Particle positions at each time point.
    """
    num_steps = int(T / dt)
    t = np.linspace(0.0, T, num_steps + 1)
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    x[0], y[0] = x0, y0

    for i in range(num_steps):
        wx, wy = np.random.normal(0.0, np.sqrt(dt), 2)
        x[i + 1] = x[i] - fx(x[i], y[i]) * dt + np.sqrt(2 * D) * wx
        y[i + 1] = y[i] - fy(x[i], y[i]) * dt + np.sqrt(2 * D) * wy

    return t, x, y