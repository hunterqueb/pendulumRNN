for i = 1:20
    m = rand(); % mass
    k = rand(); % spring constant
    c = rand() * 0.01; % damping coefficient
    F0 = rand() * 0.5; % Amplitude of the force
    omega = rand(); % Frequency of the force
    
    A = [0     1;
        -k/m -c/m];
    B = [0; 1/m];
    C = [1 0]; % Output is the position
    D = 0;
    
    sys = ss(A, B, C, D); % Create state-space model
    
    t = 0:0.01:10; % Time vector
    x0 = [1; 0]; % Initial conditions (position = 1, velocity = 0)
    u = F0 * sin(omega * t); % Sinusoidal input force
    
    [yUnforced, tUnforced, xUnforced] = lsim(sys, u, t); % Simulate the system

    [yForced, tForced, xForced] = lsim(sys, u, t, x0); % Simulate the system
end

plot(tUnforced, yUnforced); % Plot the position over time
hold on
plot(tForced, yForced); 