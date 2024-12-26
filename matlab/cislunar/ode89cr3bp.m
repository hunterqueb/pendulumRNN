function [tResult, XODE] = ode89cr3bp(x0,tEnd,numPeriods)
m_1 = 5.974E24;
m_2 = 7.348E22;
mu = m_2/(m_1 + m_2);

tf = numPeriods*tEnd;
t = [0:0.01:tf];

inteps = 1e-15; %tolerance 
opts = odeset('RelTol',inteps,'AbsTol',inteps);

[tResult,XODE] = ode89(@(t,x) CR3BPODE(t,x,mu),t,x0,opts);
end


