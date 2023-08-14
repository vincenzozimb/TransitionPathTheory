# execute the code
solve = 2

if solve == 1:
    # set parameters
    beta = 1
    dx = 0.03
    Niter = 3000
    from doublewell import run
elif solve == 2:
    from doublewell import plot
elif solve == 3:
    from doublewell import langevin
