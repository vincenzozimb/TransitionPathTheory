# run = 1 to run and analyze, run = 0 to analyze existing data 

# triple well
runtw = 1

if runtw == 1:
    # high temperature
    beta = 1.67
    from tw2d import tw_run
    from tw2d import tw_analyse
    # low temperature
    beta = 6.67
    from tw2d import tw_run
    from tw2d import tw_analyse
else:
    from tw2d import tw_analyse
