

%{
 
 This script provides the training data for the PDF prediction of a 2D
 conservative Duffing oscillator using Mamba network.

 version 2:
 - Halton nodes are included within the 3 sigma bounds.
 - A function is used to interpolate the PDF values from the RBF
 coefficients.

 Reference scripts:
 C:\Users\pu410292\CloudStation\SynologyDrive\Drive\DDDAS_2024\2D_conserv_Duff_Osc\duff2D_conserv_pdf_mamba_v1.m

%}

clear all; close all; clc;

% For random number reproducibility:
rng('default')  

% Start the clock:
tBegin = tic;

% Declare the global variable:
global k;

% Initialize a structure array:
S = struct();
Mean = struct();
Covar = struct();

% Define initial conditions and oscillator parameters.
GM = 1;
S.GM = GM;
S.omega = 1.4;
S.stiffness = 0.7;
k = S.stiffness/S.omega;

% Populate the structure array with statistical parameters:
Mean.initialPosition = 0.85; % unit: m
Mean.initialVelocity = 0;    % unit: m/s
Covar.stdInitialPosition = 0.03;
Covar.stdInitialVelocity = 0.03;
Covar.correlationInitial = 0.0;

% Generate the initial mean vector:
x0N = [Mean.initialPosition Mean.initialVelocity];

% Generate the covariance matrix:
SIGMA = [Covar.stdInitialPosition^2 Covar.correlationInitial*Covar.stdInitialPosition*Covar.stdInitialVelocity;...
                Covar.correlationInitial*Covar.stdInitialPosition*Covar.stdInitialVelocity Covar.stdInitialVelocity^2];

% Set the start and the end of propagation duration:
tStart = 0;                 % unit: s
T = 6.3756;                 % period of oscillation, unit: s (obtained using trial and error)
% tEnd = 0.749*T;   % default

% tEnd = 0.1;
% tEnd = 1;
tEnd = 4.8;  % Obtained and modified from C:\Users\pu410292\CloudStation\SynologyDrive\Drive\ASC_2023\Pugazh_Papers_2023_ASC_TwoBody_UQ_ResearchGate.pdf
% tEnd = 6;

% Set the time interval based on Hunter's request:
% delta_t = 0.001;     % unit: s

delta_t = 0.1;

% Set the pause value for figure display:
pause_val = 0.006;  % unit: s

tspan = [tStart:delta_t:tEnd];

% Set the tolerances for ode45 routine:
options = odeset('AbsTol', 1e-8, 'RelTol', 1e-8);

% Set the Mahalanobis distance from the mean:
extremal_prob = 6;
inner_prob = 3;

% Set the number of points on the extremal bounds: 
% num_extremal_pts = 120; % default
num_extremal_pts = 175;
% num_extremal_pts = 200;
% num_extremal_pts = 250; % no extra region
% num_extremal_pts = 400;
% num_extremal_pts = 500;

% Interpolation grid resolution:
inter_grid_res = 100;

% Number of dimensions with uncertianty:
un_dim_num = length(x0N);

% Obtaining the initial extremal bounds:
% [ extremal_bounds10 ] = create_extremal_bounds(x0N,SIGMA, extremal_prob,...
% num_extremal_pts);

% Obtaining the extremal bounds using the 'even_prob_span' function
% Define function handles:
funct_point_for = @duff_for_prop;
funct_point_back = @duff_back_prop;
[ extremal_bounds1, extremal_bounds10 ] = even_prob_span(x0N,...
        extremal_prob, SIGMA, num_extremal_pts, funct_point_for, [tStart tEnd], S);

% Obtain the inner bounds:
[ inner_bounds10 ] = create_extremal_bounds(x0N, SIGMA, inner_prob,...
num_extremal_pts);


% Creating a Halton point set within [0 1]^d:
% Number of Halton points:
N_halton = 130; % gives 101 points after filtering

% Instantiate a Halton object:
hp = haltonset(un_dim_num);

% Use the Halton object to create nodes in [0 1]^d:
y_set1 = net(hp, N_halton);

% Convert the Halton point set to lie within the 6 sigma extremal bounds:
rescaled_halton_set1 = zeros(N_halton, un_dim_num);
for i = 1:un_dim_num
    rescaled_halton_set1(:,i) = min(extremal_bounds10(:,i)) + ...
        (max(extremal_bounds10(:,i)) - min(extremal_bounds10(:,i)))*y_set1(:,i);
end

% Obtain the Halton nodes within the initial extremal bounds:
% parfor_use = 0;
parfor_use = 1;
if isequal(parfor_use, 1)
    % Filtering using parallel for loop:
    inPoints_halton = inhull2(rescaled_halton_set1, extremal_bounds10);
else
    % Filtering without using parallel for loop:
    inPoints_halton = inhull(rescaled_halton_set1, extremal_bounds10);
end
rbf_nodes_set1 = rescaled_halton_set1(inPoints_halton,:);

% Create a second Halton set within 3 sigma bounds:
y_set2 = net(hp, round(N_halton/2));
rbf_nodes_set2 = zeros(length(y_set2), un_dim_num);
for i = 1:un_dim_num
    rbf_nodes_set2(:,i) = min(inner_bounds10(:,i)) + ...
        (max(inner_bounds10(:,i)) - min(inner_bounds10(:,i)))*y_set2(:,i);
end

% Consolidate the two sets of RBF nodes:
rbf_nodes = [rbf_nodes_set2; rbf_nodes_set1];
nPts_rbf = length(rbf_nodes)

% Set the approximation nodes at t0:
approx_nodes_initial = rbf_nodes;

% Obtaining the PDF values at approximation nodes at t0:
initial_pdf = mvnpdf(approx_nodes_initial, x0N, SIGMA);

integ_input{1,1} = x0N;
integ_input{2,1} = extremal_bounds10;
integ_input{3,1} = rbf_nodes;

integ_output = cell(3,1);
for i = 1:size(integ_input, 1)
   for j = 1:size(integ_input{i, 1}, 1)
       [~, integ_output{i, 1}(:, :, j)] = ode45(@Duff, tspan, integ_input{i, 1}(j, :), options);
   end
end

% Provide the common string for figure 1:
fig1_str = 'state_prop_';

% Create a figure object to display state propagation:
f1 = figure('units','normalized','outerposition',[0 0 1 1]);
for kk = 1:length(tspan)
    
    figure(f1);
    plot(integ_output{1, 1}(:,1,:), integ_output{1, 1}(:,2,:), 'b.',...
        integ_output{1, 1}(kk,1,:), integ_output{1, 1}(kk,2,:), 'b*',...
        squeeze(integ_output{2, 1}(kk,1,:)), squeeze(integ_output{2, 1}(kk,2,:)), 'r*')
    grid on
    xlabel('Position (m)')
    ylabel('Velocity (m/s)')
    title(sprintf('Propagation through state space: nominal (blue) and %d \\sigma bounds (red)', extremal_prob))

%     saveas(gcf, strcat(fig1_str, num2str(kk), '.png'))
    
    % Pause each figure in the loop for visualizing:
    pause(pause_val)

end

% Make a movie:
% create_video_from_images(length(tspan), fig1_str)

% Provide the common string for figure 2:
fig2_str = 'rbf_node_distri_';

% Create a figure object to display RBF node distribution:
f2 = figure('units','normalized','outerposition',[0 0 1 1]);
for kk = 1:length(tspan)

    figure(f2);
    subplot(1,2,1)
    grid on
    plot( extremal_bounds10(:,1), extremal_bounds10(:,2), 'r*',...
        rbf_nodes(:,1), rbf_nodes(:,2), 'g.')
    xlabel('Position (m)')
    ylabel('Velocity (m/s)')
    % ax=gca;
    % ax.FontSize = 20;
    title(sprintf('2D undampled Duffing oscillator: %d \\sigma bounds (red) and RBF nodes (green) at t_0', extremal_prob))
    
    subplot(1,2,2)
    grid on
    plot( squeeze(integ_output{2, 1}(kk,1,:)), squeeze(integ_output{2, 1}(kk,2,:)), 'r*',...
        squeeze(integ_output{3, 1}(kk,1,:)), squeeze(integ_output{3, 1}(kk,2,:)), 'g.')
    xlabel('Position (m)')
    ylabel('Velocity (m/s)')
    % ax=gca;
    % ax.FontSize = 20;
    title(sprintf('2D undampled Duffing oscillator: %d \\sigma bounds (red) and RBF nodes (green) at t_f = %0.2f s', extremal_prob, tspan(kk)))

    %     saveas(gcf, strcat(fig2_str , num2str(kk), '.png'))

    % Pause each figure in the loop for visualizing:
    pause(pause_val)    
end

% Make a movie:
% create_video_from_images(length(tspan), fig2_str)

% RBF kernel:
RBFtype = 'IMQ';  % works
% RBFtype = 'MQ';
% RBFtype = 'CMQ';
% RBFtype = 'GA'; % gives +ve interpolated PDF values

% Shape parameter:
% epsilon = 0.1; 
epsilon = 0.05; % works 
% epsilon = 0.01;

% Normalizing the distribution using the maximum PDF value:
normalized_pdf = initial_pdf/max(initial_pdf);

% Initialize matrices:
normalized_coeff = zeros(nPts_rbf, length(tspan));
rescaled_approxPdf = zeros(nPts_rbf, length(tspan));
pdfApprox_error = zeros(nPts_rbf, length(tspan));

% Provide the common string for figure 3:
fig3_str = 'pdf_approx_';

% Create a figure object to display PDF approximation:
f3 = figure('units','normalized','outerposition',[0 0 1 1]);
for kk = 1:length(tspan)

    % Print the loop variable to track progress:
    kk = kk

    % Assemble the approximation nodes:
    approx_nodes = [squeeze(integ_output{3, 1}(kk,1,:)) squeeze(integ_output{3, 1}(kk,2,:))];
    
    % Approximating the normalized PDF:
    [normalized_approxPdf, normalized_coeff(:, kk)] = RBFinterp2d(approx_nodes,...
        normalized_pdf, approx_nodes, RBFtype, epsilon);
    
    % Scaling the approximated PDF to obtain the original PDF:
    rescaled_approxPdf(:, kk) = normalized_approxPdf*max(initial_pdf);
    
    % Compute the approximation error:
    pdfApprox_error(:, kk) = rescaled_approxPdf(:, kk) - initial_pdf;
    
    figure(f3);
    subplot(1,2,1)
    plot3(approx_nodes(:,1), approx_nodes(:,2), initial_pdf, 'b+')
    grid on
    xlabel('Position (m)')
    ylabel('Velocity (m/s)')
    zlabel('PDF value')
    view(-124,13)
    title(sprintf('Referenced PDF values (t_f = %0.3f s)', tspan(kk)))
    
    subplot(1,2,2)
    plot3(approx_nodes(:,1), approx_nodes(:,2), rescaled_approxPdf(:, kk), 'b+')
    grid on
    xlabel('Position (m)')
    ylabel('Velocity (m/s)')
    zlabel('PDF value')
    view(-124,13)
    title(sprintf('Approximated PDF values (t_f = %0.3f s)', tspan(kk)))
    
    % saveas(gcf, strcat(fig3_str , num2str(kk), '.png'))

    % Pause each figure in the loop for visualizing:
    pause(pause_val)

end


% Make a movie:
% create_video_from_images(length(tspan), fig3_str)

% Set the train limits:
train_limit_quarter = round(length(tspan)/4);
train_limit_half = round(length(tspan)/2);

% Set the normalizing factors:
norm_factor_quarter = max(max(normalized_coeff(:,1:train_limit_quarter)));
norm_factor_half = max(max(normalized_coeff(:,1:train_limit_half)));

% Create the coefficient matrices to give to Hunter:
double_norm_coeff_quarter = normalized_coeff/norm_factor_quarter;
double_norm_coeff_half = normalized_coeff/norm_factor_half;

% Provide the common string for figure 4:
fig4_str = 'pdf_approx_error_';

% Create a figure object to display PDF approximation error:
f4 = figure('units','normalized','outerposition',[0 0 1 1]);
for kk = 1:length(tspan)

    figure(f4);
    plot3(approx_nodes(:,1), approx_nodes(:,2), pdfApprox_error(:, kk), 'b+')
    grid on
    xlabel('Position (m)')
    ylabel('Velocity (m/s)')
    zlabel('Error')
    view(-124,13)
    title(sprintf('Error at RBF approximation nodes (t_f = %0.3f s)',tspan(kk)))
    
    % saveas(gcf, strcat(fig4_str , num2str(kk), '.png'))

    % Pause each figure in the loop for visualizing:
    pause(pause_val)

end

% Make a movie:
% create_video_from_images(length(tspan), fig4_str)

% Interpolate the PDF using the RBF coefficients:
figure_save = 1;
create_video = 1;
[finalPdf, integ_1d, cumInteg] = interpolate_PDF_with_pred_coeff_v1(integ_output,...
    norm_factor_quarter, double_norm_coeff_quarter, tspan, inter_grid_res,...
    initial_pdf, figure_save, create_video);

% Save the variables:
save('duff2D_conserv_pdf_mamba_v2.mat')












