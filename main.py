# double well
run = -1

if run == 1:
    from dw2d import dw_run
    from dw2d import dw_analyse
elif run == 0:
    from dw2d import dw_analyse


# triple well
run = 1

if run == 1:
    from tw2d import tw_run
    from tw2d import tw_analyse
elif run == 0:
    from tw2d import tw_analyse
