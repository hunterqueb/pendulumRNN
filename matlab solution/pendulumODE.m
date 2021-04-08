function dtheta = pendulumODE(t,theta)
global L;
g = 9.81;
dtheta = zeros(2,1);
% need to seperate the 2nd order to two first order equations
% dtheta1 = theta2
% dtheta2 = equation
% dtheta1 = dthetaDoubleDot
% dtheta1 refers to the acceleration
% dtheta2 referes to the velocity
dtheta(1) = theta(2);
dtheta(2) = -g/L*theta(1);
end