
import numpy as np
import matplotlib.pyplot as plt

from quebUtils.integrators import ode45


muR = 396800

DU = 6378.1 # radius of earth in km
TU = ((DU)**3 / muR)**0.5

p = 20410 # km
e = 0.8

a = p/(1-e**2)

rHEO = np.array([(p/(1+e)),0])
vHEO = np.array([0,np.sqrt(muR*((2/rHEO[0])-(1/a)))])
THEO = 2*np.pi*np.sqrt(a**3/muR)

mu = 1
r = rHEO / DU
v = vHEO * TU / DU
T = THEO / TU

J2 = 1.08263e-3

IC = np.concatenate((r,v))
pam = [mu,J2]

m_sat = 1
c_d = 2.1 #shperical model
A_sat = 1.0013 / (DU ** 2)
h_scale = 50 * 1000 / DU
rho_0 = 1.29 * 1000 ** 2 / (DU**2)

def twoBodyPert(t, y, p=pam):
    r = y[0:2]
    R = np.linalg.norm(r)
    v = y[2:4]
    v_norm = np.linalg.norm(v)

    mu = p[0]; J2 = p[1]
    dydt1 = y[2]
    dydt2 = y[3]

    factor = 1.5 * J2 * (1 / R)**2 / R**3
    j2_accel_x = factor * (1) * r[0]
    j2_accel_y = factor * (3) * r[1]

    rho = rho_0 * np.exp(-R / h_scale)  # Atmospheric density model
    drag_factor = -0.5 * (rho / m_sat) * c_d * A_sat * v_norm
    a_drag_x = drag_factor * y[2]
    a_drag_y = drag_factor *  y[3]

    a_drag_x = 0
    a_drag_y = 0
    j2_accel_x = 0
    j2_accel_y = 0

    dydt3 = -mu / R**3 * y[0] + j2_accel_x + a_drag_x
    dydt4 = -mu / R**3 * y[1] + j2_accel_y + a_drag_y

    return np.array([dydt1, dydt2,dydt3,dydt4])

numPoints = 100
numPeriods = 3
trueAnom = np.linspace(0,2*np.pi,numPoints)
# linspace f 0 to 360 

# find E from f
# 3.13b curtis
E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(trueAnom/2))

# find Me from E and e
# 3.14 curtis
Me = E - np.multiply(e,np.sin(E))

# ensure the angles lies on the the correct range from 0 to 2pi
Me[int(numPoints/2):] = Me[int(numPoints/2):] + 2*np.pi

# find t from Me and period
# 3.15 from curtis
tEval = Me/(2*np.pi) * T

# construct time eval matrix
tEvalM = np.zeros((numPoints,numPeriods))
tEvalM[:,0] = tEval

# compute for arbitrary periods
for i in range(numPeriods-1):
    tEvalM[:,i+1] = tEvalM[:,i] + T
    tEval = np.concatenate((tEval,tEvalM[1:,i+1]))

plt.figure()
plt.plot(tEval,linestyle='None',marker='s')
plt.xlabel('True Anomoly (rad)')
plt.ylabel('Time (nondim)')
plt.show()

t,y = ode45(twoBodyPert,[0,numPeriods*T],IC,tEval)
# t,y_eqi = ode45(twoBodyPert,[0,T],IC)

plt.figure()
plt.plot(y[:,0],y[:,1],linestyle='None',marker='s')
# plt.plot(y_eqi[:,0],y_eqi[:,1],linestyle='None',marker='o')

plt.show()