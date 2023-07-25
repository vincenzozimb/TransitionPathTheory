import numpy as np
import matplotlib.pyplot as plt


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
        x: numpy array
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



# Define parameters
x0, y0 = 0.0, 0.0  # Initial positions (x, y)
D = 1.0           # Diffusion coefficient
dt = 0.01         # Time step for numerical integration
T = 10.0          # Total time for integration

# Define deterministic forces for the x and y directions (e.g., harmonic forces)
k_x, k_y = 0.1, 0.1
f_x = lambda x, y: -k_x * x
f_y = lambda x, y: -k_y * y

# Solve the 2D Langevin equation
t, x, y = overdamped_langevin(x0, y0, f_x, f_y, D, dt, T)

# Plot the results
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Overdamped Langevin Equation')
plt.grid(True)
plt.show()