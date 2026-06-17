import numpy as np
from scipy.integrate import solve_ivp



def sigmoid(v, e0=2.5, v0=6.0, r=0.56):
    """
    Population voltage --> firing rate transformation that applies to each of the three neuronal populations.
    e0 = half the max firing rate
    r = sigmoid time constant, determines sigmoid steepness
    v0 = threshold, center of sigmoid
    Default values taken from standard formulation.
    """

    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


# to work with the IVP solver, this must accept time and state vector (t, y) and return a derivative vector
def jansen_rit(t, y, p=120.0, A=3.25, B=22.0, a=100.0, b=50.0, C=135.0):
    """
    y0 = pyramidal PSP
    y1 = excitatory PSP input to pyramidal cells
    y2 = inhibitory PSP input to pyramidal cells
    y3 = dy0/dt
    y4 = dy1/dt
    y5 = dy2/dt

    t = current time
    y = current state vector, holds all six state variables
    p = external input
    A, B = excitatory, inhibitory gain
    a, b = excitatory, inhibitory time constant
    C = connectivity strength

    Default values taken from standard model.
    """
    y0, y1, y2, y3, y4, y5 = y

    # internal coupling scaling constants for populations
    C1 = C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C

    dy0 = y3
    dy1 = y4
    dy2 = y5

    # y1 - y2 is the net synaptic input to the pyramidal cells
    dy3 = A * a * sigmoid(y1 - y2) - 2 * a * y3 - (a**2) * y0

    # excitatory interneuron synaptic eq
    dy4 = A * a * (p + C2 * sigmoid(C1 * y0)) - 2 * a * y4 - (a**2) * y1

    # inhibitory interneuron synaptic eq
    dy5 = B * b * (C4 * sigmoid(C3 * y0)) - 2 * b * y5 - (b**2) * y2

    # return vector of derivatives for IVP solver
    return [dy0, dy1, dy2, dy3, dy4, dy5]


def solve_jr(t_end=2000, sf=1000, params=None, y0_init=None):
    """
    sf = 1000 # sampling frequency 1000Hz
    t_end = how long the sim will run for
    initial value problem solver
    given the initial condition and ODEs, it will approximate y(t) for times after t0
    returns an OdeResult object with fields
          sol.t = time points associated w solutions
          sol. y = solution matrix n x m
          column i of sol.y corresponds to time sol.t[i]
          row j is yj's approximate value
          this means indexing works like:
              sol.y[1] gives the array of y1 values at every time point in sol.t
    """
    t_start = 0.0

    # linspace takes (start, end, # points)
    t_eval = np.linspace(
        t_start, t_end, int((t_end - t_start) * sf) + 1
    )  # points = intervals + 1

    # all six state variables start at zero
    y0_init = np.zeros(6)

    def jr_rhs(t, y):
        return jansen_rit(t, y, **params)

    sol = solve_ivp(
        fun=jr_rhs,
        t_span=(t_start, t_end),  # integrate over this time range
        y0=y0_init,
        t_eval=t_eval,  # the time points where we want it to sample
    )

    return sol


def simulate_observation(params=None, t_end=2000, sf=1000):
    sol = solve_jr(t_end=t_end, sf=sf, params=params)
    t = sol.t
    eeg_proxy = sol.y[1] - sol.y[2]
    return t, eeg_proxy
