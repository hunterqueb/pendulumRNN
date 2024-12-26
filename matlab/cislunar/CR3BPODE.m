function dydt = CR3BPODE(~, Y, mu)
% Solve the CR3BP in nondimensional coordinates.
%
% The state vector is Y, with the first two components as the
% position of m, and the second two components its velocity.
%
% The solution is parameterized on mu, the mass ratio.
%
% Arguments:
% t: current time
% Y: current state vector
% mu: mass ratio (default value is mu = 0.012277471)
%
% Returns:
% dydt: derivative vector

if nargin < 3
    mu = 0.012277471; % Default value of mu
end

% Get the position and velocity from the solution vector
x = Y(1);
y = Y(2);
z = Y(3);
xdot = Y(4);
ydot = Y(5);
zdot = Y(6);

r1 = sqrt((x + mu)^2 + y^2 + z^2);
r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);

dydt = [xdot;
        ydot;
        zdot;
        2 * ydot + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3;
        -2 * xdot + y - (1 - mu) * y / r1^3 - mu * y / r2^3;
        -(1-mu) * z / r1^3 - mu * z / r2^3;];

end
