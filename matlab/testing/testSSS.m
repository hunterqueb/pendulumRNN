m = 1; % mass
k = 1; % spring constant
c = 0.1; % damping coefficient
F0 = 0.5; % Amplitude of the force
omega = 1; % Frequency of the force

A = [0     1;
    -k/m -c/m];
B = [0; 1/m];
C = [1 0]; % Output is the position
D = 0;

sys = ss(A, B, C, D); % Create state-space model

t = 0:0.1:10; % Time vector
x0 = [1; 0]; % Initial conditions (position = 1, velocity = 0)
u = F0 * sin(omega * t); % Sinusoidal input force

[y, t, x] = lsim(sys, u, t, x0); % Simulate the system

plot(t, y); % Plot the position over time
