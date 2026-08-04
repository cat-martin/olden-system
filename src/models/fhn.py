import numpy as np
from scipy.integrate import solve_ivp
from src.util.config import base_fhn_params, fhn_duration

# rhs of the ivp solver
def fhn_rhs(t, state, a, b, c, I):
    '''rhs of the ivp solver for FitzHugh-Nagumo'''
    V, w = state

    # fhn equations
    dV_dt = V * (a - V) * (V - 1) - w + I
    dw_dt = b * V - c * w

    return [dV_dt, dw_dt]


def simulate_fhn(params=base_fhn_params, end=fhn_duration):
    '''Actually runs the solver and integrates, returns time and membrane variable traces for an individual unit using 'params' '''
    a, b, c, I = (
        params["a"],
        params["b"],
        params["c"],
        params["I"],
    )

    # ivp solver settings
    init_state = [0.0, 0.0]
    t_span = [0.0, end]
    t_points = np.linspace(0.0, end, int(end*10) + 1)

    # solver
    sol = solve_ivp(
        fhn_rhs,
        t_span,
        init_state,
        args=(a, b, c, I),
        t_eval=t_points,
    )

    # make sure solver finishes
    if not sol.success:
        raise RuntimeError(f'FHN solver failed - {sol.message}')

    # unpack soln into state vars over time
    V = sol.y[0]

    return t_points, V