import desolver as de
import desolver.backend as D

def rhs(t, state, k, m, **kwargs):
    return D.array([[0.0, 1.0], [-k/m,  0.0]])@state

y_init = D.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=1e-9, atol=1e-9, constants=dict(k=1.0, m=1.0))

print(a)

a.integrate()

print(a)

print("If the integration was successful and correct, a[0].y and a[-1].y should be near identical.")
print("a[0].y  = {}".format(a[0].y))
print("a[-1].y = {}".format(a[-1].y))

print("Maximum difference from initial state after one oscillation cycle: {}".format(D.max(D.abs(a[0].y-a[-1].y))))