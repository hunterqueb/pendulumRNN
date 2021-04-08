global L;
L = 0.5;

theta0=[100 0]*pi/180;
tspan = [0 10];

[t, theta] = ode45(@pendulumODE,tspan,theta0);

plot(t,theta(:,1))

figure
plot(t,theta(:,2))