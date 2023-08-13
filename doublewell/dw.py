# define the potential
def potential(x, y):
    return 5/2 * (x**2 - 1)**2 + 5 * y**2


# gradient of the potential
def V_partialx(x, y):
    return 10 * x * (x**2 - 1)

def V_partialy(x, y):
    return 10 * y