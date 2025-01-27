import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
import time
from qutils.integrators import myRK4

def uniformRandomPointOnSphere(radius = (10/180)*np.pi):
    """Generates a random point on the surface of a unit sphere."""

    

    # Generate a random point within a cube
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-1, 1)

    # Normalize the point to project it onto the sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    while norm > 1:  # Ensure the point is inside the sphere
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        norm = np.sqrt(x**2 + y**2 + z**2)

    x = x / norm * radius
    y = y / norm * radius
    z = z / norm * radius

    return x, y, z

def plotSphere(radius = (10/180)*np.pi):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)

    x = np.outer(np.sin(theta), np.cos(phi)) * radius
    y = np.outer(np.sin(theta), np.sin(phi))* radius
    z = np.outer(np.cos(theta), np.ones_like(phi))* radius

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    # ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    return fig,ax

def nominalEulerAngleMotionDerivative(t, x,p=None):  

    s3t = np.sin(3*t)
    c3t = np.cos(3*t)
    s5t = np.sin(5*t)
    c5t = np.cos(5*t)

    dydt1 = 3*c3t*c5t - 5*s3t*s5t
    dydt2 = 5.5*c5t
    dydt3 = 0.5 * (c5t*5*(0.1+s3t)**3 + 9*c3t*(0.1+s3t)**2*s5t)
    # dydt3 = 0.5 * 9 * s5t*c3t*(s3t+0.1)**2

    return dydt1, dydt2,dydt3

def calcNonlinearIndex(currentSTM,nominalSTM):
    return np.linalg.norm(currentSTM - nominalSTM) / np.linalg.norm(nominalSTM)

def nominalEulerAngleMotion(t):  

    s3t = np.sin(3*t)
    c3t = np.cos(3*t)
    s5t = np.sin(5*t)
    c5t = np.cos(5*t)

    y1 = s3t*c5t
    y2 = 1.1*s5t
    y3 = 0.5 * (s5t * (0.1+s3t)**3)

    return y1,y2,y3

def calcNominalSTM(phi,theta,psi):

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    STM = np.array(([0,spsi,cpsi],[0,ctheta*cpsi,-ctheta*spsi],[ctheta,stheta*spsi,stheta*cpsi]))/ctheta

    return STM


def getAngularMotion(t):
    dphi,dtheta,dpsi = nominalEulerAngleMotionDerivative(t,None)
    phi,theta,psi = nominalEulerAngleMotion(t)
    STM = calcNominalSTM(phi,theta,psi)

    dx = np.array([dphi,dtheta,dpsi]).reshape(3,)

    omegas = np.linalg.inv(STM) @ dx

    return omegas

def calcLinDepartSTM(omegas,t):
    phi,theta,psi = nominalEulerAngleMotion(t)
    
    w1=omegas[0]
    w2=omegas[1]
    w3=omegas[2]

    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    ttheta = np.tan(theta)

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    

    A = np.array(([0,(spsi*w1+cpsi*w3)*ttheta,cpsi*w2-spsi*w3],[0,0,-spsi*w2-cpsi*w3],[0,(spsi*w2+cpsi*w3)/ctheta,(cpsi*w2-spsi*w3)*stheta]))/ctheta

    return A

def calcNewLinDepartState(t,dphi,dtheta,dpsi):
    omegas = getAngularMotion(t)
    A = calcLinDepartSTM(omegas,t)

    STM = sp.linalg.expm(A)

    dx = np.array((dphi,dtheta,dpsi)).reshape(3,1)

    x = STM @ dx

    return x[0],x[1],x[2]

if __name__ == "__main__":
    
    radius = (10/180) * np.pi
    print(radius)
    np.random.seed(int(time.time() * 1000) % 2**32)

    N = 300
    timeSteps = 100
    t0 = 0; tf = 25
    t = np.linspace(t0,tf,timeSteps)

    # nominalTrajRK4 = myRK4(nominalEulerAngleMotionDerivative,np.array((0,0,0)),[t0,tf],paramaters=None)

    sphereFig,sphereAxis = plotSphere()

    nonlinearIndex = np.array([[None] * N ] * len(t))

    for i in range(N):
        xi,yi,zi = uniformRandomPointOnSphere()
        assert xi**2 + yi**2 + zi**2 - radius ** 2 < 0.000001
        sphereAxis.scatter(xi, yi, zi, c='r', zorder=10)
        for j in range(len(t)):

            xn,yn,zn = nominalEulerAngleMotion(t[j])
        
            nonlinearIndex[j,i] = calcNonlinearIndex(calcLinDepartSTM(getAngularMotion(t[j]),t[j]),calcNominalSTM(xn,yn,zn))
    v = np.max(nonlinearIndex,axis=0)
    # print(np.max(nonlinearIndex))
    # print(nonlinearIndex)
    # plt.show()


