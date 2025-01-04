m_1 = 5.974E24;
m_2 = 7.348E22;
mu = m_2/(m_1 + m_2);

x0 = [1.42280126e+00  3.54236466e-20 -4.60924421e-24  7.33744863e-12 -1.01690547e+00  1.57609568e-24];
x0 = [ 9.0453898750573813E-1	-3.0042855182227924E-26	1.4388186844294218E-1	-8.5656563732450135E-15	-4.9801575824700677E-2	-1.9332247649544646E-14	];
tEnd = 3.7265552310265724E+0;
numPeriods = 5;

tf = numPeriods*tEnd;
t = [0:0.001:tf];

inteps = 1e-15; %tolerance 
opts = odeset('RelTol',inteps,'AbsTol',inteps);

tic
[tResult,XODE] = ode89(@(t,x) CR3BPODE(t,x,mu),t,x0,opts);
toc

plot3(XODE(:,1),XODE(:,2),XODE(:,3))


save 'CR3BP_butterfly_1080.mat' tResult XODE