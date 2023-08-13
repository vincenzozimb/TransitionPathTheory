import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

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
q = []
m = []
J = [[], []]

# Open the CSV file for reading
with open('doublewell/data/data.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    # Read and skip the header row
    header = next(csv_reader)
    # Iterate through each row in the CSV file
    for row in csv_reader:
        x_val, y_val, V_val, P_val, q_val, m_val, Jx_val, Jy_val = map(float, row)
        # Append values to their respective lists
        X.append(x_val)
        Y.append(y_val)
        V.append(V_val)
        prob.append(P_val)
        q.append(q_val)
        m.append(m_val)
        J[0].append(Jx_val)
        J[1].append(Jy_val)

# Reshape the loaded data back to their original meshgrid shape
X = np.array(X).reshape((Nx, Ny))  
Y = np.array(Y).reshape((Nx, Ny))  
V = np.array(V).reshape((Nx, Ny))  
prob = np.array(prob).reshape((Nx, Ny))
q = np.array(q).reshape((Nx, Ny)) 
m = np.array(m).reshape((Nx, Ny))  
J = [np.array(J[0]).reshape((Nx, Ny)), np.array(J[1]).reshape((Nx, Ny))]

# Create folder for images, if it does not already exist
if not os.path.exists('doublewell/images'):
   os.makedirs('doublewell/images')

# Visualize the potential and the Giggs distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, V, levels=15, cmap='viridis')
plt.colorbar(label='Potential')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potential Energy')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, prob, levels=15, cmap='viridis')
plt.colorbar(label='Probability density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gibbs distribution')

plt.tight_layout()
plt.savefig("doublewell/images/p&g.png")

# plot the committor
metastable_boolean = V < Vmax
fig = plt.subplots(figsize=(8, 6))
contourf = plt.contourf(X, Y, q, levels=10, cmap='viridis')
highlighted_levels = np.array([-1])  # Value that highlights the region
highlighted_contour = np.where(metastable_boolean, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=1)
plt.colorbar(contourf)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Forward committor')
plt.savefig("doublewell/images/c_tpt.png")

# plot m
fig, ax = plt.subplots(figsize=(8, 6))
contourf = ax.contourf(X, Y, m, levels=10, cmap='viridis')
highlighted_contour = np.where(metastable_boolean, highlighted_levels, np.nan)
plt.contourf(X, Y, highlighted_contour, colors='white', alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel of the transition path density')
plt.savefig("doublewell/images/m_tpt.png")

# plot of the vector field
magnitude = np.sqrt(J[0]**2 + J[1]**2)
magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
plt.figure(figsize=(8, 6))
plt.streamplot(X.T, Y.T, J[0].T, J[1].T, density=1.5, color=magnitude_normalized.T, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.contourf(X, Y, highlighted_contour, colors='lightgray', alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transition Path Current')
# plt.gca().set_facecolor('lightgray')  # Set background color
plt.tight_layout()
plt.savefig("doublewell/images/J_tpt.png")


# subsample_factor = 2
# X_plot = X[::subsample_factor, ::subsample_factor]
# Y_plot = Y[::subsample_factor, ::subsample_factor]
# Jx_plot = J[0][::subsample_factor, ::subsample_factor]
# Jy_plot = J[1][::subsample_factor, ::subsample_factor]

# plt.figure(figsize=(8, 6))
# plt.quiver(X_plot, Y_plot, Jx_plot, Jy_plot, angles='xy', scale_units='xy', scale=0.1, color='darkblue', alpha=0.7, width=0.005, headwidth=4)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Transition path current')
# plt.gca().set_facecolor('lightgray')  # Set background color
# plt.savefig("doublewell/J_tpt.png")

plt.show()