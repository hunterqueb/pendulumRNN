import numpy as np
# ──────────────────────────────────────────────────────────────────────────────
#  Runge–Kutta–Fehlberg 4(5) coefficients (Butcher tableau)
# ──────────────────────────────────────────────────────────────────────────────
_A = np.array([
    [],                                                          # row 0 (unused)
    [1/4],
    [3/32,           9/32],
    [1932/2197,     -7200/2197,      7296/2197],
    [439/216,           -8,          3680/513,     -845/4104],
    [-8/27,              2,         -3544/2565,    1859/4104,    -11/40],
], dtype=object)

_C = np.array([0, 1/4, 3/8, 12/13, 1, 1/2], dtype=float)

# 5th- and 4th-order output weights
_B5 = np.array([16/135,       0,  6656/12825,   28561/56430,   -9/50,     2/55])
_B4 = np.array([25/216,       0, 1408/2565,     2197/4104,     -1/5,      0   ])

# Difference gives the error estimator coefficients
_E  = _B5 - _B4
# ──────────────────────────────────────────────────────────────────────────────


def rkf45(fun, tspan, y0, t_eval=None, rtol=1e-8, atol=1e-8):
    """
    Adaptive Runge–Kutta–Fehlberg 4(5) ODE solver (Matlab-style interface).

    Parameters
    ----------
    fun : callable(t, y) → dy/dt
        Vector field of the ODE system.
    tspan : (float, float)
        Integration interval (t0, tf).  Direction (forward/backward) is inferred.
    y0 : array_like, shape (n,)
        Initial state.
    t_eval : array_like or None
        Times at which the solution is returned.  If None, all internal steps
        are returned.
    rtol, atol : float
        Relative and absolute error tolerances (per component).
    h_max, h_min : float
        Maximum / minimum step size allowed.

    Returns
    -------
    t : ndarray, shape (m,)
    y : ndarray, shape (m, n)
        Solution sampled at `t`.
    """
    
    t0, tf = float(tspan[0]), float(tspan[1])
    direction = np.sign(tf - t0) or 1.0
    t_curr, y_curr = t0, np.atleast_1d(y0).astype(float)
    n = y_curr.size

    h_max=(tf-t0)/2
    h_min=1e-12 * max(abs(t0), 1.0)


    # Helper for error norm (Shampine 1997 §4.1)
    def _error_norm(err, y_new):
        scale = atol + rtol * np.maximum(np.abs(y_curr), np.abs(y_new))
        return np.linalg.norm(err / scale) / np.sqrt(n)

    # Choose an initial step (first derivative-based heuristic)
    f0 = np.atleast_1d(fun(t_curr, y_curr))
    h = 0.01 * direction * (abs(tf - t0) + 1e-14)  # heuristic
    if np.all(f0 == 0):
        h = 1e-6 * direction
    else:
        h = 0.01 * direction * np.min(atol / (np.abs(f0) + 1e-14))**0.5
    h = np.clip(abs(h), 1e-12, h_max) * direction

    # Output storage
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        if direction > 0:
            assert np.all((t_eval >= t0) & (t_eval <= tf)), \
                "`t_eval` values must lie inside `tspan`"
        else:
            assert np.all((t_eval <= t0) & (t_eval >= tf)), \
                "`t_eval` values must lie inside `tspan`"
        next_eval_idx = 0
        t_out = []
        y_out = []
    else:
        t_out = [t_curr]
        y_out = [y_curr.copy()]

    # Main integration loop --------------------------------------------------
    while (direction > 0 and t_curr < tf) or (direction < 0 and t_curr > tf):

        # Clip step to not over-shoot tf
        h = direction * min(abs(h), abs(tf - t_curr))
        if abs(h) < h_min:
            raise RuntimeError("Step size underflow.")

        # Compute the 6 stages (k1 … k6)
        k = np.empty((6, n))
        k[0] = f0
        for i in range(1, 6):
            ti = t_curr + _C[i] * h
            yi = y_curr + h * sum(_A[i][j] * k[j] for j in range(i))
            k[i] = fun(ti, yi)

        # Fifth- and fourth-order estimates
        y5 = y_curr + h * np.dot(_B5, k)
        y4 = y_curr + h * np.dot(_B4, k)
        err = y5 - y4
        err_norm = _error_norm(err, y5)

        if err_norm <= 1.0:        # Accept step
            t_curr += h
            y_curr = y5
            f0 = k[-1]             # reuse last stage derivative

            # Record solution at requested points
            if t_eval is None:
                t_out.append(t_curr)
                y_out.append(y_curr.copy())
            else:
                # Add all t_eval points we just passed (using 5th-order poly)
                while (next_eval_idx < len(t_eval) and
                       (direction > 0 and t_eval[next_eval_idx] <= t_curr) or
                       (direction < 0 and t_eval[next_eval_idx] >= t_curr)):
                    tau = (t_eval[next_eval_idx] - (t_curr - h)) / h  # in [0,1]
                    # 5th-order dense output (uploaded from Dormand & Prince 1980)
                    y_dense = (
                        y_curr
                        - h * np.dot(_E, k) * ((1 - tau) ** 5)
                    )  # simple polynomial; error ~ O(h^5)
                    t_out.append(t_eval[next_eval_idx])
                    y_out.append(y_dense.copy())
                    next_eval_idx += 1

        # Step-size controller (Dormand–Prince PI controller, α=0.2)
        safety = 0.9
        if err_norm == 0:
            h *= 5.0
        else:
            h *= safety * err_norm ** (-0.2)
        h = np.clip(abs(h), h_min, h_max) * direction

    t = np.array(t_out)
    y = np.stack(y_out, axis=0)
    return t, y



# ──────────────────────────────────────────────────────────────────────────────
# Dormand–Prince 5(4) tableau  (same as SciPy’s RK45)
# ──────────────────────────────────────────────────────────────────────────────
_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

_A = [
    [],
    [1/5],
    [3/40,         9/40],
    [44/45,       -56/15,        32/9],
    [19372/6561,  -25360/2187,   64448/6561,    -212/729],
    [9017/3168,   -355/33,       46732/5247,      49/176,   -5103/18656],
    [35/384,        0,           500/1113,       125/192,  -2187/6784,  11/84]
]

_B  = np.array([35/384,       0,  500/1113, 125/192, -2187/6784, 11/84, 0])   # 5-th order
_B4 = np.array([5179/57600,   0,  7571/16695, 393/640, -92097/339200,
                187/2100, 1/40])                                                # 4-th order
_E  = _B - _B4                                                                   # error estimator


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def _norm_inf(x):
    return np.linalg.norm(x, ord=np.inf)


def _initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
    """Hairer–Wanner initial-step heuristic (SciPy’s choice)."""
    scale = atol + rtol * np.abs(y0)
    d0 = _norm_inf(y0 / scale)
    d1 = _norm_inf(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + direction * h0 * f0
    f1 = fun(t0 + direction * h0, y1)
    d2 = _norm_inf((f1 - f0) / scale) / h0

    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

    return min(100 * h0, h1)


# ---------------------------------------------------------------------------
# Dense-output coefficient tensor  (shape 7 × 6)
# Rows 0…6 → k0…k6;  columns → θ¹ θ² θ³ θ⁴ θ⁵ θ⁶
# ---------------------------------------------------------------------------
_DENSE_COEFFS = np.array([
    [  1.        ,  -183./64. ,    37./12. ,   -145./128.,     0.,         0. ],
    [  0.        ,     0.     ,     0.     ,      0.     ,     0.,         0. ],
    [  0.        ,  1500./371., -1000./159.,  1000./371.,     0.,         0. ],
    [  0.        ,  -125./32. ,   125./12. ,   -375./64.,     0.,         0. ],
    [  0.        ,  9477./3392.,  -729./106., 25515./6784.,    0.,         0. ],
    [  0.        ,   -11./7.  ,    11./3.  ,   -55./28. ,     0.,         0. ],
    [  0.        ,     1./2.  ,     -1.    ,     1./2.  ,     0.,         0. ],
])

def _dense_output(theta, y0, h, k):
    """
    5th-order dense interpolant identical to SciPy's _DOPRI5DenseOutput.

    Parameters
    ----------
    theta : float or ndarray in [0,1]
    y0    : ndarray, state at left end of step
    h     : float,   step size
    k     : ndarray shape (7, n) with stages k0 … k6
    """
    theta = np.asarray(theta)
    th2   = theta * theta
    th3   = th2 * theta
    th4   = th3 * theta
    th5   = th4 * theta

    poly = (
        k[0]*(1
               + theta*(-183./64.
               + theta*(  37./12.
               + theta*( -145./128.))))
      + k[2]*(th2*( 1500./371.
               + theta*( -1000./159.
               + theta*(  1000./371.))))
      + k[3]*(th2*( -125./32.
               + theta*(   125./12.
               + theta*(  -375./64.))))
      + k[4]*(th2*( 9477./3392.
               + theta*(  -729./106.
               + theta*( 25515./6784.))))
      + k[5]*(th2*(  -11./7.
               + theta*(   11./3.
               + theta*(   -55./28.))))
      + k[6]*(th2*(   1./2.
               + theta*(     -1.
               + theta*(    1./2.))))
    )

    return y0 + h*poly

# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────
def ode45(fun, tspan, y0, *, t_eval=None, rtol=1e-6, atol=1e-9,
          first_step=None, max_step=np.inf, vectorized=False):
    """
    Dormand–Prince RK45 integrator that mirrors SciPy’s algorithm but
    returns MATLAB-style `(t, y)`.

    Parameters
    ----------
    fun : callable(t, y) -> ndarray
    tspan : (t0, tf)
    y0    : array_like, shape (n,)
    t_eval, rtol, atol, first_step, max_step : see SciPy
    vectorized : ignored (pure-Python path)

    Returns
    -------
    t : ndarray (m,)
    y : ndarray (m, n)
    """
    t0, tf = map(float, tspan)
    direction = np.sign(tf - t0) or 1.0

    y0 = np.atleast_1d(y0).astype(float)
    n = y0.size

    t = t0
    y = y0
    f = fun(t, y)

    order = 5
    if first_step is None:
        h = _initial_step(fun, t, y, f, direction, order, rtol, atol)
    else:
        h = direction * abs(first_step)
    h = min(abs(h), max_step) * direction

    # Output containers -------------------------------------------------------
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        if direction > 0:
            assert np.all((t_eval >= t0) & (t_eval <= tf))
        else:
            assert np.all((t_eval <= t0) & (t_eval >= tf))
        tout, yout = [], []
        next_eval = 0
        if t_eval[0] == t:
            tout.append(t); yout.append(y.copy()); next_eval = 1
    else:
        tout, yout = [t], [y.copy()]

    # PI controller parameters (SciPy’s values)
    SAFETY, MIN_FACTOR, MAX_FACTOR, BETA = 0.9, 0.2, 10.0, 0.04
    err_prev = 1.0         # initialise to 1

    # Main adaptive loop ------------------------------------------------------
    while (direction > 0 and t < tf) or (direction < 0 and t > tf):

        h = direction * min(abs(h), abs(tf - t), max_step)

        if t_eval is not None and next_eval < len(t_eval):
            h = direction * min(abs(h), abs(t_eval[next_eval] - t))

        # ─ Compute stages k0 … k6 ───────────────────────────────────────
        k = np.empty((7, n))
        k[0] = f
        for i in range(1, 7):
            ti = t + _C[i] * h
            yi = y + h * sum(_A[i][j] * k[j] for j in range(i))
            k[i] = fun(ti, yi)

        # 5-th-order solution & error
        y5  = y + h * np.dot(_B,  k)
        err =      h * np.dot(_E, k)

        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = _norm_inf(err / scale)

        # ─ Accept / reject step ─────────────────────────────────────────
        if err_norm <= 1.0:            # accept
            t_new = t + h
            y_new = y5
            f_new = k[-1].copy()       # derivative at the new point

            if t_eval is not None:
                while (next_eval < len(t_eval) and
                       ((direction > 0 and t_eval[next_eval] <= t_new) or
                        (direction < 0 and t_eval[next_eval] >= t_new))):
                    theta = (t_eval[next_eval] - t) / h
                    y_interp = _dense_output(theta, y, h, k)
                    tout.append(t_eval[next_eval]); yout.append(y_interp)
                    next_eval += 1
            else:
                tout.append(t_new); yout.append(y_new.copy())

            t, y, f = t_new, y_new, f_new

        # ─ PI step-size update ──────────────────────────────────────────
        if err_norm == 0.0:
            factor = MAX_FACTOR
        else:
            factor = SAFETY * err_norm**(-0.2) * err_prev**BETA
            factor = np.clip(factor, MIN_FACTOR, MAX_FACTOR)
        h *= factor
        err_prev = max(err_norm, 1e-16)        # avoid zero in next power

    return np.asarray(tout), np.stack(yout, axis=0)






from scipy.integrate import solve_ivp
from qutils.tictoc import timer
# ────────────────────────────────────────────────────────────────
# Earth constants (WGS-84)
# ────────────────────────────────────────────────────────────────
mu = 3.986004418e14        # m³/s²  – Earth GM
R  = 6_378_137.0           # m      – mean equatorial radius

# ────────────────────────────────────────────────────────────────
# Two-body + J2 acceleration (same algebra as AstroForge)
# ────────────────────────────────────────────────────────────────
def twoBodyJ2(t, y, mu=mu, R=R):
    x = y[:3]
    v = y[3:]

    J2 = 4.84165368e-4 * np.sqrt(5)          # √5·J₂ (matches AstroForge)

    M2 = J2 * np.diag((0.5, 0.5, -1.0))
    r  = np.sqrt(x @ x)

    F0  = -mu * x / r**3                                           # monopole
    acc = (mu * R**2 / r**5) * (-5 * x * (x @ M2 @ x) / r**2
                                + 2 * M2 @ x) + F0

    ydot = np.empty_like(y)
    ydot[:3] = v
    ydot[3:] = acc
    return ydot


# ────────────────────────────────────────────────────────────────
# Initial conditions: 700-km circular LEO, equatorial
# ────────────────────────────────────────────────────────────────
alt = 700_000.0                       # altitude [m]
a   = R + alt                         # semimajor axis [m] (≈ radius, circular)
v_circ = np.sqrt(mu / a)              # orbital speed for circular orbit

y0 = np.array([a, 0, 0,       # position (x, y, z) in ECI
               0, v_circ, 0]) # velocity (vx,vy,vz)

T_orb = 2 * np.pi * np.sqrt(a**3 / mu)  # Kepler period ≈ 98 min
t_span = (0.0, T_orb)                   # propagate one orbit
t_eval = np.linspace(0, T_orb, 1001)    # sample every ≈6 s

# --- propagate with *no* dense output ---------------------------------
t_my, y_my = ode45(twoBodyJ2, (0, T_orb), y0,   # <-- no t_eval argument
                   rtol=1e-9, atol=1e-12)

# ask SciPy to evaluate *exactly* on our mesh
sol        = solve_ivp(twoBodyJ2, (0, T_orb), y0,
                       method='RK45', t_eval=t_my,
                       rtol=1e-9, atol=1e-12)

print("max step-end diff :", np.max(np.abs(sol.y.T - y_my)))

# ────────────────────────────────────────────────────────────────
# 1) Integrate with custom ode45 (imported from your module)
# ────────────────────────────────────────────────────────────────
time = timer()
t_me, y_me = ode45(twoBodyJ2, t_span, y0, t_eval=t_eval,
                   rtol=1e-9, atol=1e-12)
time.toc()
# ────────────────────────────────────────────────────────────────
# 2) Integrate with SciPy’s RK45
# ────────────────────────────────────────────────────────────────
time = timer()
scipy_sol = solve_ivp(twoBodyJ2, t_span, y0, method="RK45",
                      rtol=1e-9, atol=1e-12, t_eval=t_eval)
time.toc()
# ────────────────────────────────────────────────────────────────
# 3) Compare results
# ────────────────────────────────────────────────────────────────
err = np.abs(y_me - scipy_sol.y.T)
print(f"Max position error  : {err[:, :3].max():.3e} m")
print(f"Max velocity error  : {err[:, 3:].max():.3e} m/s\n")

print(f"ode45 steps accepted: {len(t_me)-1}")
print(f"SciPy steps accepted: {len(scipy_sol.t)-1}")
print("max interpolated diff :", np.max(np.abs(y_my - sol.y.T)))
