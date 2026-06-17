import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# rhs of the ivp solver
def fhn_rhs(t, state, a, b, c, I):
    V, w = state

    # fhn equations
    dV_dt = V * (a - V) * (V - 1) - w + I
    dw_dt = b * V - c * w

    return [dV_dt, dw_dt]

# baseline params
a = 0.1
b = 0.01
c = 0.02
I = 0.1

# ivp solver settings
init_state = [0.0, 0.0]
t_span = [0.0, 500.0]
t_points = np.linspace(0.0, 500.0, 5000)

# solver
sol = solve_ivp(
    fhn_rhs,
    t_span,
    init_state,
    args=(a, b, c, I),
    t_eval=t_points,
)

# unpack soln into state vars over time
V = sol.y[0]
w = sol.y[1]

# transient check, chop off first 100 secs
transient = 100.0
mask = sol.t >= transient

# baseline descriptors
V_steady = V[mask]
w_steady = w[mask]
V_mean = np.mean(V_steady)
w_mean = np.mean(w_steady)
V_std = np.std(V_steady)

print("Mean V: ", V_mean)
print("Std V: ", V_std)

# dominant frequency
dt = sol.t[1] - sol.t[0]
freqs = np.fft.rfftfreq(len(V_steady), d=dt)   # create possible frequencies
spectrum = np.abs(np.fft.rfft(V_steady - V_mean))  # remove vertical offset, return strength of frequency contributions
dominant_idx = np.argmax(spectrum[1:]) + 1   # ignores frequency 0, returns largest of remaining values, +1 since we skip frq 0
dominant_freq = freqs[dominant_idx]
print("Dominant Frequency: ", dominant_freq)

period = 1.0 / dominant_freq
print("Period: ", period)

# plotting
plt.figure()
plt.plot(freqs[1:], spectrum[1:])

plt.xlim(0, dominant_freq * 5)

plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of V")

plt.show()


plt.plot(sol.t[mask], V[mask], label="V")
plt.plot(sol.t[mask], w[mask], label="w")

plt.xlabel("Time")
plt.ylabel("State Val")
plt.title("FHN State Evolution")
plt.legend()
plt.show()