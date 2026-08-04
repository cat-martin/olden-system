# 4 cycles is 2.08 sec
# 8 cycles is 3.76 sec
# 12 cycles is 5.44 sec
jr_duration = 5.44
base_jr_params = {
    'p': 120.0, 
    'A': 3.25, 
    'B': 22.0, 
    'a': 100.0, 
    'b': 50.0, 
    'C': 135.0,
    'v0': 6.0,
}

# 4 cycles is 540 TU
# 8 cycles is 980 TU
# 12 cycles is 1420 TU
fhn_duration = 1420
base_fhn_params = {
    'a': -0.1,
    'b': 0.01,
    'c': 0.02,
    'I': 0.1,
}

half_widths = {
    'a': 0.17,
    'tau': 45.0,
    'q': 0.015,
    'v0': 0.1
}

simple_h_vals = [0.0, 0.25, 0.5, 0.75, 1.0]