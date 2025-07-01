import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np


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


_DENSE_P = np.array([
    [1,           -183/64,    37/12,   -145/128,       0,          0],
    [0,                 0,         0,          0,       0,          0],
    [0,          1500/371,  -1000/159, 1000/371,       0,          0],
    [0,           -125/32,     125/12,  -375/64,       0,          0],
    [0,        9477/3392,    -729/106, 25515/6784,     0,          0],
    [0,             -11/7,       11/3,   -55/28,       0,          0],
    [0,               1/2,        -1,      1/2,       0,          0]
])   # shape (7, 6)  — columns are θ, θ², θ³, θ⁴, θ⁵, θ⁶ (last two zero here)

def _dense_output(theta, y0, h, k):
    """
    Accurate to 5th order, identical to SciPy’s _DOPRI5DenseOutput.
    theta may be scalar or ndarray, 0 ≤ theta ≤ 1.
    """
    # powers: [θ, θ², θ³, θ⁴, θ⁵, θ⁶]
    t_powers = np.vstack([theta ** i for i in range(1, 6+1)])   # shape (6, …)

    # Dot product P · powers gives coefficients b_i(θ) for each stage k_i
    b = (_DENSE_P[:, :6] @ t_powers).T        # shape (…, 7)
    return y0 + h * np.einsum('...i,i...->...', b, k)


# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────
def spy_ode45(fun, tspan, y0, t_eval=None, rtol=1e-8, atol=1e-8,
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

        # # ───── trim step so we land exactly on the next t_eval ─────
        # if t_eval is not None and next_eval < len(t_eval):
        #     h = direction * min(abs(h), abs(t_eval[next_eval] - t))

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


from qutils.integrators import ode45,direct_ode45

from memory_profiler import profile
plotOn = True

problemDim = 4 

m1 = 1
m2 = m1
l1 = 1
l2 = l1
g = 9.81
parameters = np.array([m1,m2,l1,l2,g])

def doubleLinPends(t,y):
    k=1;m=1
    dydt1=-k/m * y[1]
    dydt2=y[0]
    dydt3=-k/m * y[3]
    dydt4=y[2]
    return np.array((dydt1,dydt2,dydt3,dydt4))

def doublePendulumODE(t,y,p=parameters):
    # p = [m1,m2,l1,l2,g]
    m1 = p[0]
    m2 = p[1]
    l1 = p[2]
    l2 = p[3]
    g = p[4]

    theta1 = y[0]
    theta2 = y[2]

    dydt1 = y[1] #theta1dot

    dydt2 = (m2*g*np.sin(theta2)*np.cos(theta1-theta2) - m2*np.sin(theta1-theta2)*(l1*y[1]**2*np.cos(theta1-theta2) + l2*y[3]**2)
            - (m1+m2)*g*np.sin(theta1)) / l1 / (m1 + m2*np.sin(theta1-theta2)**2) #theta1ddot

    dydt3 = y[3] #theta2dot

    dydt4 = ((m1+m2)*(l1*y[1]**2*np.sin(theta1-theta2) - g*np.sin(theta2) + g*np.sin(theta1)*np.cos(theta1-theta2))
            + m2*l2*y[3]**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / l2 / (m1 + m2*np.sin(theta1-theta2)**2) #theta2ddot

    return np.array((dydt1,dydt2,dydt3,dydt4))

theta1_0 = np.radians(80)
theta2_0 = np.radians(135)
thetadot1_0 = np.radians(-1)
thetadot2_0 = np.radians(0.7)


theta1_0 = 1
theta2_0 = 0
thetadot1_0 = -1
thetadot2_0 = 0.7


initialConditions = np.array([theta1_0,thetadot1_0,theta2_0,thetadot2_0],dtype=np.float64)
initialConditions = np.radians(np.random.uniform(-180, 180, (problemDim,)))

tStart = 0
tEnd = 20
tSpan = np.array([tStart,tEnd])
dt = 0.01
tSpanExplicit = np.linspace(tStart,tEnd,int(tEnd / dt))

funcPointer = doubleLinPends

t, numericResult = ode45(funcPointer,tSpan,initialConditions,tSpanExplicit)
direct_t, direct_numericResult = direct_ode45(funcPointer,tSpan,initialConditions,tSpanExplicit)
sp_t, sp_numericResult = spy_ode45(funcPointer,tSpan,initialConditions,tSpanExplicit)

if plotOn is True:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(numericResult[:,0],numericResult[:,2],'r',label = "Truth")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')


    plt.subplot(2, 1, 2)
    plt.plot(numericResult[:,1],numericResult[:,3],'r',label = "Truth")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(direct_numericResult[:,0],direct_numericResult[:,2],'r',label = "Truth")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')


    plt.subplot(2, 1, 2)
    plt.plot(direct_numericResult[:,1],direct_numericResult[:,3],'r',label = "Truth")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(sp_numericResult[:,0],sp_numericResult[:,2],'r',label = "Truth")
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 1 Dot')
    plt.axis('equal')


    plt.subplot(2, 1, 2)
    plt.plot(sp_numericResult[:,1],sp_numericResult[:,3],'r',label = "Truth")
    plt.xlabel('Theta 2')
    plt.ylabel('Theta 2 Dot')
    plt.axis('equal')
    plt.legend()

    print(numericResult.shape)
    print(direct_numericResult.shape)
    print(sp_numericResult.shape)

    err = direct_numericResult - numericResult
    err1 = sp_numericResult - numericResult
    plt.figure()
    plt.plot(t,err)
    plt.title("scipy v rkf45")
    
    plt.figure()
    plt.plot(t,err1)
    plt.title('scipy v dp rk45')
    plt.show()


