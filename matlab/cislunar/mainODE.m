function [tResult, XODE] = mainODE()
m_1 = 5.974E24;
m_2 = 7.348E22;
mu = m_2/(m_1 + m_2);

x0 = [1.42280126e+00  3.54236466e-20 -4.60924421e-24  7.33744863e-12 -1.01690547e+00  1.57609568e-24];

tEnd = 19.117149553060344;
numPeriods = 5;

tf = numPeriods*tEnd;
t = [0:0.01:tf];

inteps = 1e-15; %tolerance 
opts = odeset('RelTol',inteps,'AbsTol',inteps);

tic
[tResult,XODE] = ode89(@(t,x) CR3BPODE(t,x,mu),t,x0,opts);
toc
end


