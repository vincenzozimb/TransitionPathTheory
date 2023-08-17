# execute the code
solve = 4

if solve == 1:
    # set parameters
    beta = 1
    dx = 0.03
    Niter = 3000
    from doublewell import run
elif solve == 2:
    from doublewell import plot
elif solve == 3:
    # set parameters
    dt = 0.01           # Time step for numerical integration
    T = 10              # Total time for integration
    D = 1               # Diffusion coefficient
    from doublewell import langevin
elif solve == 4:
    # set parameters
    beta = 10
    h = 0.001
    Niter = 5000
    from triplewell import tw
elif solve == 5:
    from test import prova