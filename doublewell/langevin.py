import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from doublewell import dw
from functions.func import overdamped_langevin


# Import parameters
from main import dt, T, D

file_path = 'doublewell/data/parameters.pkl'
with open(file_path, 'rb') as file:
    parameters = pickle.load(file)

Nx = parameters['Nx']
Ny = parameters['Ny']
Vmax = parameters['Vmax']


# Initialize empty lists to store data columns
X = []
Y = []
V = []


# Open the CSV file for reading
with open('doublewell/data/data.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    # Read and skip the header row
    header = next(csv_reader)
    # Iterate through each row in the CSV file
    for row in csv_reader:
        x_val, y_val, V_val = map(float, row[:3])
        # Append values to their respective lists
        X.append(x_val)
        Y.append(y_val)
        V.append(V_val)


# Reshape the loaded data back to their original meshgrid shape
X = np.array(X).reshape((Nx, Ny))  
Y = np.array(Y).reshape((Nx, Ny))  
V = np.array(V).reshape((Nx, Ny))  


# Create folder for images, if it does not already exist
if not os.path.exists('doublewell/images'):
   os.makedirs('doublewell/images')


# Initial positions (x, y)
x0, y0 = -1.0, 0.0 


# Define deterministic forces for the x and y directions
f_x = lambda x, y: dw.V_partialx(x, y)
f_y = lambda x, y: dw.V_partialy(x, y)


# Solve the 2D Langevin equation
t, x, y = overdamped_langevin(x0, y0, f_x, f_y, D, dt, T)


# # Plot the trajectory
# plt.contourf(X, Y, V, levels=15, cmap='coolwarm')
# plt.colorbar(label='Energy')
# plt.plot(x_reac, y_reac, linestyle='-', color='blue', label='Trajectory')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Energy Landscape with Trajectory')
# plt.legend()
# plt.show()


# indicator functions of the metastable states
def in_reac(x, y):
    return x < 0 and dw.potential(x, y) < Vmax  # condition for reactant set 

def in_prod(x, y):
    return x > 0 and dw.potential(x, y) < Vmax  # condition for product set 


# first and last passage time functions
def t_plus(t0, t, x, y, R, P):
    """
    Calculate the first passage time to the reactant set or to the product set starting from time t.

    Parameters:
        t0: Independent time variable.
        t: Array of times of the trajectory data.
        x: Array of x-coordinate trajectory data.
        y: Array of y-coordinate trajectory data.
        R: A function that returns True if a point is in set R.
        P: A function that returns True if a point is in set P.

    Returns:
        float: The first passage time to R or P starting from time t, or np.inf if not reached.
    """
    for n in range(len(t)):
        if t[n] < t0:
            continue
        if R(x[n], y[n]) or P(x[n], y[n]):
            return t[n]
    return np.inf # check this (should be fine)
    # for improvement (add also in t_minus)
    # for ti, xi, yi in zip(t, x, y):
    #     if ti < t0:
    #         continue
    #     if R(xi, yi) or P(xi, yi):
    #         return ti
    # return np.inf

def t_minus(t0, t, x, y, R, P):
    """
    Calculate the last passage time to the reactant set or to the product set starting from time t.

    Parameters:
        t0: Independent time variable.
        t: Array of times of the trajectory data.
        x: Array of x-coordinate trajectory data.
        y: Array of y-coordinate trajectory data.
        R: A function that returns True if a point is in set R.
        P: A function that returns True if a point is in set P.

    Returns:
        float: The last passage time to R or P ending at time t, or -np.inf if not reached.
    """
    for n in reversed(range(len(t))):
        if t[n] > t0:
            continue
        if R(x[n], y[n]) or P(x[n], y[n]):
            return t[n]
    return -np.inf # check this (should be fine)




