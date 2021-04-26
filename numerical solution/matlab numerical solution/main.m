global L;
L = 0.5;

theta0=[80 0]*pi/180;
tspan = [0 20];

[t, theta] = ode45(@pendulumODEFriction,tspan,theta0);

figure
plot(t,theta(:,1))

% figure
% plot(t,theta(:,2))