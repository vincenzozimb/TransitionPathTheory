import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from doublewell import dw
from functions.func import overdamped_langevin

# Import parameters
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
prob = []


# Open the CSV file for reading
with open('doublewell/data/data.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    # Read and skip the header row
    header = next(csv_reader)
    # Iterate through each row in the CSV file
    for row in csv_reader:
        x_val, y_val, V_val, P_val = map(float, row[:4])
        # Append values to their respective lists
        X.append(x_val)
        Y.append(y_val)
        V.append(V_val)
        prob.append(P_val)


# Reshape the loaded data back to their original meshgrid shape
X = np.array(X).reshape((Nx, Ny))  
Y = np.array(Y).reshape((Nx, Ny))  
V = np.array(V).reshape((Nx, Ny))  
prob = np.array(prob).reshape((Nx, Ny))


# Create folder for images, if it does not already exist
if not os.path.exists('doublewell/images'):
   os.makedirs('doublewell/images')


# Define parameters
x0, y0 = -1.0, 0.0  # Initial positions (x, y)
D = 1               # Diffusion coefficient
dt = 0.01           # Time step for numerical integration
T = 10              # Total time for integration


# Define deterministic forces for the x and y directions
f_x = lambda x, y: dw.V_partialx(x, y)
f_y = lambda x, y: dw.V_partialy(x, y)


# Solve the 2D Langevin equation
t, x, y = overdamped_langevin(x0, y0, f_x, f_y, D, dt, T)

# Plot the results
plt.contourf(X, Y, V, levels=15, cmap='coolwarm')
plt.colorbar(label='Energy')
plt.plot(x, y, linestyle='-', color='blue', label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Energy Landscape with Trajectory')
plt.legend()
plt.show()
