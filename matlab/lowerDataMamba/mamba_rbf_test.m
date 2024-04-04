
%{

This script provides the training data for MAMBA neural network to learn
the numerical integration and PDF approximation of the 2D damped Duffing
oscillator. 

Reference script: C:\Users\pu410292\CloudStation\SynologyDrive\Drive\HigherDimensionalApproximation\RBF-PDF\ROIpopulation\gridPop2DdampedDuffOsc_v2.m

%}

% Start from a clean workspace:
clear all; close all; clc;

% Including the function pathways:
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\LM_MATLABscripts\ChebFuncs')
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\LM_MATLABscripts\ProbFuncs')
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\LM_MATLABscripts\ForceFuncs')
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\LM_MATLABscripts\Utilities')

% To include the create_extremal_bounds(), inhull2():
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\OPA-main\FireOPALvalidation\fireopal_ucf_data')

% To include RBF related functions:
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\HigherDimensionalApproximation\RBF-PDF')

% For random number reproducibility:
rng('default') 

% Set the tolerances for ode45 routine:
options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

% Initializing global variables and some data structures:
global k c;
S = struct();
Mean = struct();
Covar = struct();

%%

% Define initial conditions and oscillator parameters:
GM = 1; 
S.GM = GM;
S.omega = 1.4;
S.stiffness = 0.7;

% Nonlinear stiffness coefficient: (N/m)
k = S.stiffness/S.omega;
% k = 0.6;
% k = 0.7;

% Damping coefficient: (Ns/m)
c = 0.5;

% Mean.initialPosition = 0.85; % unit: m
Mean.initialPosition = 0.8;
Mean.initialVelocity = 0;    % unit: m/s
Covar.stdInitialPosition = 0.03;
Covar.stdInitialVelocity = 0.03;
Covar.correlationInitial = 0.0;
x0N = [Mean.initialPosition Mean.initialVelocity];
SIGMA = [Covar.stdInitialPosition^2 Covar.correlationInitial*Covar.stdInitialPosition*Covar.stdInitialVelocity;...
                Covar.correlationInitial*Covar.stdInitialPosition*Covar.stdInitialVelocity Covar.stdInitialVelocity^2];

% Set the time span for the problem: (unit: s)
% Starting time:
tStart = 0;     

% Period of the oscillator:
% T = 6.3756;                 % period of oscillation, unit: s (obtained using trial and error)
T = 10;

% Ending time:
tEnd = 0.749*T;

% Time resolution:
delta_t = 0.1;
tspan = [tStart:delta_t:tEnd];

% Propagate the damped Duffing oscillator:
[tX, Xdamped] = ode45(@dampedDuff, tspan, x0N, options);

% Propagate the undamped Duffing oscillator for comparison:
[tUndamped, Xundamped] = ode45(@Duff, tspan, x0N, options);

% Number of dimensions with uncertainty:
un_dim_num = length(x0N);

% Set the PDF truncation level:
extremal_prob = 6;

% Set the number of points on the extremal bounds:
num_extremal_pts = 150;

% Initialize a 3D array for extremal bounds:
extremal_bounds = zeros(num_extremal_pts, un_dim_num, length(tspan));

% Initialize an array for the extremal bound area:
bound_area = zeros(length(tspan),1);

% Obtaining the extremal bounds at t0:
extremal_bounds(:,:,1) = create_extremal_bounds(x0N,SIGMA, extremal_prob,...
    round(num_extremal_pts));

% Propagate the extremal bound point to tf:
for i = 1:num_extremal_pts
    [tX, damp_dummy] = ode45(@dampedDuff, tspan, extremal_bounds(i,:,1), options);
    extremal_bounds(i,:,:) = damp_dummy';
end

% Animation like plot to visualize the extremal bound propagation:
figure
for i = 1:length(tspan)
    bound_area(i,1) = polyarea(extremal_bounds(:,1,i),extremal_bounds(:,2,i));
    txt = ['area: ' num2str(bound_area(i,1))];
    plot(Xdamped(:,1), Xdamped(:,2), 'r.', Xundamped(:,1), Xundamped(:,2), 'b.',...
        extremal_bounds(:,1,i),extremal_bounds(:,2,i),'r*')
    text(0, 0.6, txt)
    grid on
    pause(0.1)
end
xlabel('Position (m)')
ylabel('Velocity (m/s)')
legend('damped', 'undamped', '6 \sigma bound')
title('Propagation of the 6 \sigma bounds through state space')

% Creating a Halton point set within [0 1]^d:
% Number of Halton points:
% N_halton = 500;
N_halton = 800;
hp = haltonset(un_dim_num);
y = net(hp, N_halton);

% Convert the Halton point set to lie within the 6 sigma extremal bounds:
rescaled_halton = zeros(N_halton, un_dim_num);
for i = 1:un_dim_num
    rescaled_halton(:,i) = min(extremal_bounds(:,i,1)) + ...
        (max(extremal_bounds(:,i,1)) - min(extremal_bounds(:,i,1)))*y(:,i);
end

% Obtain the Halton nodes within the initial extremal bounds:
% parfor_use = 0;
parfor_use = 1;
if isequal(parfor_use, 1)
    % Filtering using parallel for loop:
    inPoints_halton = inhull2(rescaled_halton, extremal_bounds(:,:,1));
else
    % Filtering without using parallel for loop:
    inPoints_halton = inhull(rescaled_halton, extremal_bounds(:,:,1));
end
nPts_rbf = sum(inPoints_halton)

% Initialize a 3D array for RBF nodes:
rbf_nodes = zeros(nPts_rbf, un_dim_num, length(tspan));
rbf_nodes(:,:,1) = rescaled_halton(inPoints_halton,:);

% Propagate the RBF nodes to tf:
for i = 1:nPts_rbf
    [tX, damp_dummy] = ode45(@dampedDuff, tspan, rbf_nodes(i,:,1), options);
    rbf_nodes(i,:,:) = damp_dummy';
end

figure
subplot(1,2,1)
plot(Xdamped(1,1), Xdamped(1,2), 'b.', extremal_bounds(:,1,1), extremal_bounds(:,2,1),'r*',...
    rbf_nodes(:,1,1), rbf_nodes(:,2,1), 'g.')
grid on
hold on;
title('6 \sigma extremal bounds(t_0)')
xlabel('Position (m)')
ylabel('Velocity (m/s)')
legend('Nominal state',  '6 \sigma bound', 'RBF nodes')

subplot(1,2,2)
plot(Xdamped(end,1), Xdamped(end,2), 'b.', extremal_bounds(:,1,end), extremal_bounds(:,2,end),'r*',...
    rbf_nodes(:,1,end), rbf_nodes(:,2,end), 'g.')
grid on
hold on;
title('6 \sigma extremal bounds(t_f)')
xlabel('Position (m)')
ylabel('Velocity (m/s)')
legend('Nominal state',  '6 \sigma bound', 'RBF nodes')

% Animation like plot to visualize the extremal bound propagation:
figure
for i = 1:length(tspan)
    plot(extremal_bounds(:,1,i),extremal_bounds(:,2,i),'r*',...
        rbf_nodes(:,1,i), rbf_nodes(:,2,i), 'g.')
    grid on
    pause(0.1)
end
xlabel('Position (m)')
ylabel('Velocity (m/s)')
legend('6 \sigma bound', 'RBF nodes')
title('Propagation of the RBF nodes through state space')

%% RBF coefficients estimation:

% Set the type of radial basis used:
RBFtype = 'CMQ';  % CMQ = Coupled Multi-Quadric
% RBFtype = 'MQ';  % MQ = Multi-Quadric
% RBFtype = 'IMQ';

% Set the shape parameter:
% epsilon = 0.1;
epsilon = 0.05;

% Set the interpolation grid resolution:
% grid_res = 20;
grid_res = 25;

%%% At t0:
% Collect the RBF nodes at t0:
rbf_nodes_t0 = [extremal_bounds(:,:,1);rbf_nodes(:,:,1)];

% Sample the PDF at RBF nodes:
pdf_t0 = mvnpdf(rbf_nodes_t0, x0N, SIGMA);

% Normalizing the PDF so that it lies between 0 and 1:
normalized_pdf_t0 = pdf_t0/max(pdf_t0);

% Obtain the RBF coefficients at t0:
[normalized_approxPdf_t0, normalized_coeff_t0] = RBFinterp2d(rbf_nodes_t0, normalized_pdf_t0, rbf_nodes_t0, RBFtype, epsilon);

% Obtain the regular PDF:
approxPdf_t0 = normalized_approxPdf_t0*max(pdf_t0);

% Create the evaluation grid and filter it using the extremal bounds:
[testx_t0, testy_t0] = ndgrid(linspace(min(rbf_nodes_t0(:,1)), max(rbf_nodes_t0(:,1)), grid_res), ...
    linspace(min(rbf_nodes_t0(:,2)), max(rbf_nodes_t0(:,2)), grid_res));
eval_grid_t0 = [testx_t0(:) testy_t0(:)];
indices_t0 = inhull2(eval_grid_t0, extremal_bounds(:,:,1));
in_eval_grid_t0 = eval_grid_t0(indices_t0, :);

% Interpolate the PDF onto the filtered evaluation grid:
normalized_dummy_pdf_t0 = RBFeval2d(rbf_nodes_t0, in_eval_grid_t0, normalized_coeff_t0, RBFtype, epsilon);
dummy_pdf_t0 = normalized_dummy_pdf_t0*max(pdf_t0);
inter_pdf_t0 = zeros(length(eval_grid_t0),1);
inter_pdf_t0(indices_t0,1) = dummy_pdf_t0;

% Compute the marginal PDF and the cumulative integral:
integ_1d_t0 = trapz(testy_t0(1,:), reshape(inter_pdf_t0, size(testx_t0)), 2);
cumInteg_t0 = zeros(length(integ_1d_t0),1);
for i = 2:length(cumInteg_t0)
    cumInteg_t0(i,1) = trapz(testx_t0(1:i,1), integ_1d_t0(1:i));
end
I_t0 = trapz(testx_t0(:,1), integ_1d_t0)

figure
subplot(2,2,1)
plot(rbf_nodes_t0(:,1), rbf_nodes_t0(:,2),  'b.')
hold on
grid on
plot3(rbf_nodes_t0(:,1), rbf_nodes_t0(:,2), normalized_approxPdf_t0, 'r.')
view(-31,14)
xlabel('Position (m)')
ylabel('Velocity (m/s)')
zlabel('PDF value')
title('2D Duffing oscillator:approximated PDF at t_0')

subplot(2,2,2)
plot(eval_grid_t0(:,1), eval_grid_t0(:,2),  'b.')
hold on
grid on
plot3(eval_grid_t0(:,1), eval_grid_t0(:,2), inter_pdf_t0, 'r.')
view(-31,14)
xlabel('Position (m)')
ylabel('Velocity (m/s)')
zlabel('PDF value')
title('2D Duffing oscillator: interpolated PDF at t_0')

subplot(2,2,3)
plot(testx_t0(:,1), integ_1d_t0)
grid on
xlabel('Position (m)')
ylabel('Marginal PDF')
title('1D marginal PDF at t0')

subplot(2,2,4)
plot(testx_t0(:,1), cumInteg_t0)
grid on
xlabel('Position (m)')
ylabel('Cumulative probability')
title('Cumulative probability at t0')


%%% At t1 ( = t0 + delta_t):
% Collect the RBF nodes at t1:
rbf_nodes_t1 = [extremal_bounds(:,:,2);rbf_nodes(:,:,2)];

% Scale the PDF value using the area contraction:
pdf_t1 = (bound_area(1,1)/bound_area(2,1))*pdf_t0;

% Normalizing the PDF so that it lies between 0 and 1:
normalized_pdf_t1 = pdf_t1/max(pdf_t1);

% Obtain the RBF coefficients at t1:
[normalized_approxPdf_t1, normalized_coeff_t1] = RBFinterp2d(rbf_nodes_t1, normalized_pdf_t1, rbf_nodes_t1, RBFtype, epsilon);

% Obtain the regular PDF:
approxPdf_t1 = normalized_approxPdf_t1*max(pdf_t1);

% Create the evaluation grid and filter it using the extremal bounds:
[testx_t1, testy_t1] = ndgrid(linspace(min(rbf_nodes_t1(:,1)), max(rbf_nodes_t1(:,1)), grid_res), ...
    linspace(min(rbf_nodes_t1(:,2)), max(rbf_nodes_t1(:,2)), grid_res));
eval_grid_t1 = [testx_t1(:) testy_t1(:)];
indices_t1 = inhull2(eval_grid_t1, extremal_bounds(:,:,2));
in_eval_grid_t1 = eval_grid_t1(indices_t1, :);

% Interpolate the PDF onto the filtered evaluation grid:
normalized_dummy_pdf_t1 = RBFeval2d(rbf_nodes_t1, in_eval_grid_t1, normalized_coeff_t1, RBFtype, epsilon);
dummy_pdf_t1 = normalized_dummy_pdf_t1*max(pdf_t1);
inter_pdf_t1 = zeros(length(eval_grid_t1),1);
inter_pdf_t1(indices_t1,1) = dummy_pdf_t1;

% Compute the marginal PDF and the cumulative integral:
integ_1d_t1 = trapz(testy_t1(1,:), reshape(inter_pdf_t1, size(testx_t1)), 2);
cumInteg_t1 = zeros(length(integ_1d_t1),1);
for i = 2:length(cumInteg_t1)
    cumInteg_t1(i,1) = trapz(testx_t1(1:i,1), integ_1d_t1(1:i));
end
I_t1 = trapz(testx_t1(:,1), integ_1d_t1)

figure
subplot(2,2,1)
plot(rbf_nodes_t1(:,1), rbf_nodes_t1(:,2),  'b.')
hold on
grid on
plot3(rbf_nodes_t1(:,1), rbf_nodes_t1(:,2), approxPdf_t1, 'r.')
view(-31,14)
xlabel('Position (m)')
ylabel('Velocity (m/s)')
zlabel('PDF value')
title('2D Duffing oscillator:approximated PDF at t_1')

subplot(2,2,2)
plot(eval_grid_t1(:,1), eval_grid_t1(:,2),  'b.')
hold on
grid on
plot3(eval_grid_t1(:,1), eval_grid_t1(:,2), inter_pdf_t1, 'r.')
view(-31,14)
xlabel('Position (m)')
ylabel('Velocity (m/s)')
zlabel('PDF value')
title('2D Duffing oscillator: interpolated PDF at t_1')

subplot(2,2,3)
plot(testx_t1(:,1), integ_1d_t1)
grid on
xlabel('Position (m)')
ylabel('Marginal PDF')
title('1D marginal PDF at t1')

subplot(2,2,4)
plot(testx_t1(:,1), cumInteg_t1)
grid on
xlabel('Position (m)')
ylabel('Cumulative probability')
title('Cumulative probability at t1')

%%% At tf:
% Collect the RBF nodes at tf:
rbf_nodes_tf = [extremal_bounds(:,:,end);rbf_nodes(:,:,end)];

% Scale the PDF value using the area contraction:
pdf_tf = (bound_area(1,1)/bound_area(end,1))*pdf_t0;

% Normalizing the PDF so that it lies between 0 and 1:
normalized_pdf_tf = pdf_tf/max(pdf_tf);

% Obtain the RBF coefficients at tf:
[normalized_approxPdf_tf, normalized_coeff_tf] = RBFinterp2d(rbf_nodes_tf, normalized_pdf_tf, rbf_nodes_tf, RBFtype, epsilon);

% Obtain the regular PDF:
approxPdf_tf = normalized_approxPdf_tf*max(pdf_tf);

% Create the evaluation grid and filter it using the extremal bounds:
[testx_tf, testy_tf] = ndgrid(linspace(min(rbf_nodes_tf(:,1)), max(rbf_nodes_tf(:,1)), grid_res), ...
    linspace(min(rbf_nodes_tf(:,2)), max(rbf_nodes_tf(:,2)), grid_res));
eval_grid_tf = [testx_tf(:) testy_tf(:)];
indices_tf = inhull2(eval_grid_tf, extremal_bounds(:,:,end));
in_eval_grid_tf = eval_grid_tf(indices_tf, :);

% Interpolate the PDF onto the filtered evaluation grid:
normalized_dummy_pdf_tf = RBFeval2d(rbf_nodes_tf, in_eval_grid_tf, normalized_coeff_tf, RBFtype, epsilon);
dummy_pdf_tf = normalized_dummy_pdf_tf*max(pdf_tf);
inter_pdf_tf = zeros(length(eval_grid_tf),1);
inter_pdf_tf(indices_tf,1) = dummy_pdf_tf;

% Compute the marginal PDF and the cumulative integral:
integ_1d_tf = trapz(testy_tf(1,:), reshape(inter_pdf_tf, size(testx_tf)), 2);
cumInteg_tf = zeros(length(integ_1d_tf),1);
for i = 2:length(cumInteg_tf)
    cumInteg_tf(i,1) = trapz(testx_tf(1:i,1), integ_1d_tf(1:i));
end
I_tf = trapz(testx_tf(:,1), integ_1d_tf)

figure
subplot(2,2,1)
plot(rbf_nodes_tf(:,1), rbf_nodes_tf(:,2),  'b.')
hold on
grid on
plot3(rbf_nodes_tf(:,1), rbf_nodes_tf(:,2), approxPdf_tf, 'r.')
view(-31,14)
xlabel('Position (m)')
ylabel('Velocity (m/s)')
zlabel('PDF value')
title('2D Duffing oscillator:approximated PDF at t_f')

subplot(2,2,2)
plot(eval_grid_tf(:,1), eval_grid_tf(:,2),  'b.')
hold on
grid on
plot3(eval_grid_tf(:,1), eval_grid_tf(:,2), inter_pdf_tf, 'r.')
view(-31,14)
xlabel('Position (m)')
ylabel('Velocity (m/s)')
zlabel('PDF value')
title('2D Duffing oscillator: interpolated PDF at t_f')

subplot(2,2,3)
plot(testx_tf(:,1), integ_1d_tf)
grid on
xlabel('Position (m)')
ylabel('Marginal PDF')
title('1D marginal PDF at tf')

subplot(2,2,4)
plot(testx_tf(:,1), cumInteg_tf)
grid on
xlabel('Position (m)')
ylabel('Cumulative probability')
title('Cumulative probability at tf')

matrix_t0 = [rbf_nodes_t0 normalized_pdf_t0 normalized_coeff_t0];
matrix_t1 = [rbf_nodes_t1 normalized_pdf_t1 normalized_coeff_t1];
matrix_tf = [rbf_nodes_tf normalized_pdf_tf normalized_coeff_tf];

save('mamba_rbf_test.mat')

