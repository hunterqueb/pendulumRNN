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
