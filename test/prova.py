import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 5)
x = t**2

for i in range(len(t)):
    print(i, t[i], x[i])

for i in reversed(range(len(t))):
    print(i, t[i], x[i])

