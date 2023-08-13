# set parameters
beta = 1
dx = 0.03
Niter = 3000


# execute the code
solve = 3

if solve == 1:
    from doublewell import run
elif solve == 2:
    from doublewell import plot
elif solve == 3:
    from doublewell import langevin
