import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from doublewell import dw
from functions.func import overdamped_langevin


# Import parameters
from main import dt, T, D
from doublewell.dw import Vmax

file_path = 'doublewell/data/parameters.pkl'
with open(file_path, 'rb') as file:
    parameters = pickle.load(file)

Nx = parameters['Nx']
Ny = parameters['Ny']


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


# indicator functions
def one_T(x0, y0):
    if dw.in_reac(x0, y0) or dw.in_prod(x0, y0):
        return 0.0
    else:
        return 1.0
    
def one_R(t0):
    tm = dw.t_minus(t0, t, x, y)
    x0 = x[tm]
    y0 = y[tm]
    if dw.in_reac(x0, y0):
        return 1.0
    else:
        return 0.0

def one_P(t0):
    tp = dw.t_plus(t0, t, x, y)
    x0 = x[tp]
    y0 = y[tp]
    if dw.in_prod(x0, y0):
        return 1.0
    else:
        return 0.0


# g-function (Dirac delta)
h = 0.03
def g_delta(x0, y0, xpar, ypar):
    if np.abs(x0 - xpar) < h and np.abs(y0 - ypar) < h:
        return 1.0
    else:
        return 0.0


# numerator integral
num = np.zeros((Nx, Ny))



