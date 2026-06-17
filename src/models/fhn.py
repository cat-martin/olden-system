"""
Minimal FitzHugh–Nagumo baseline implementation.

Equations:
    dV/dt = V(a - V)(V - 1) - w + I
    dw/dt = bV - cw
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class FHNParameters:
    a: float
    b: float
    c: float
    I: float

    @classmethod
    def from_recovery_parameters(
        cls,
        a: float,
        tau_w: float,
        kappa_w: float,
        I: float,
    ):
        if tau_w <= 0:
            raise ValueError("tau_w must be greater than zero.")

        c = 1.0 / tau_w
        b = kappa_w * c
        return cls(a=a, b=b, c=c, I=I)

    @property
    def tau_w(self) -> float:
        if self.c <= 0:
            raise ValueError("c must be greater than zero to define tau_w.")
        return 1.0 / self.c

    @property
    def kappa_w(self) -> float:
        if self.c == 0:
            raise ValueError("c must be nonzero to define kappa_w.")
        return self.b / self.c


def fhn_rhs(
    t: float,
    state: np.ndarray,
    params: FHNParameters,
    input_function: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    V, w = state
    applied_input = params.I if input_function is None else input_function(t)

    dV_dt = V * (params.a - V) * (V - 1.0) - w + applied_input
    dw_dt = params.b * V - params.c * w

    return np.array([dV_dt, dw_dt], dtype=float)


def simulate_fhn(
    params: FHNParameters,
    initial_state: Tuple[float, float] = (0.0, 0.0),
    t_start: float = 0.0,
    t_end: float = 500.0,
    dt: float = 0.05,
    input_function: Optional[Callable[[float], float]] = None,
):
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start.")
    if dt <= 0:
        raise ValueError("dt must be greater than zero.")

    t_eval = np.arange(t_start, t_end + dt, dt)

    solution = solve_ivp(
        fun=lambda t, y: fhn_rhs(t, y, params, input_function),
        t_span=(t_start, t_end),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError("FHN integration failed: " + solution.message)

    return solution.t, solution.y[0], solution.y[1]


def plot_fhn(
    t: np.ndarray,
    V: np.ndarray,
    w: np.ndarray,
    transient: float = 0.0,
) -> None:
    mask = t >= transient

    plt.figure(figsize=(10, 5))
    plt.plot(t[mask], V[mask], label="V")
    plt.plot(t[mask], w[mask], label="w", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("FitzHugh–Nagumo baseline simulation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Temporary example values only.
    # Replace these with the canonical values you decide to validate.
    params = FHNParameters(
        a=0.1,
        b=0.01,
        c=0.02,
        I=0.1,
    )

    print("Raw parameters:")
    print("  a =", params.a)
    print("  b =", params.b)
    print("  c =", params.c)
    print("  I =", params.I)
    print()
    print("Derived recovery quantities:")
    print("  tau_w =", params.tau_w)
    print("  kappa_w =", params.kappa_w)

    t, V, w = simulate_fhn(
        params=params,
        initial_state=(0.0, 0.0),
        t_end=500.0,
        dt=0.05,
    )

    plot_fhn(t, V, w, transient=100.0)


if __name__ == "__main__":
    main()
