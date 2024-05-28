function [finalPdf, integ_1d, cumInteg] = interpolate_PDF_with_pred_coeff_v1(integ_output,...
    norm_factor, pred_coeff, tspan, inter_grid_res, initial_pdf, figure_save, create_video)

% This function interpolates the PDF using the RBF approximation
% coefficients predicted by Mamba. It operates on the delta_t set by
% Pugazh. Please check with him for the suitable delta_t.

% Inputs:
% integ_output: a 3 x 1 cell array of which the last two cells carry the
% extremal bounds for each time instant and the approximation coefficients.
% norm_factor: a 1 x 1 scalar that was used to normalize the coefficients
% while they were generated for training the Mamba network.
% pred_coeff: a m x n matrix of coefficients predicted by Mamba network
% tuned by Hunter. Here, m is the number of coefficients at each time
% instant and n is the number of time steps.
% tspan: a 1 x n vector of time steps at which the coefficients are
% predicted.
% inter_grid_res: a 1 x 1 scalar that sets the resolution of the
% interpolating grid.
% initial_pdf: a m x 1 vector of initial PDF values at t0.
% figure_save: a 1 x 1 logical scalar that decides if the figures should be
% saved or not. 1 = yes, 0 = no.
% create_vide0: a 1 x 1 logical scalar that decides if a video should be
% created using the saved files. 1 = yes, 0 = no.

% Outputs:
% finalPdf: a n x 1 cell array of PDFs for each time instant.
% integ_1d: a n x 1 cell array of 1D marginal PDFs for each time instant.
% cumInteg: a n x 1 cell array of cumulative integrals for each time instant.

    %%% PLEASE DON'T CHANGE THIS %%%%%
    % RBF kernel:
    RBFtype = 'IMQ';  % works
    
    % Shape parameter:
    epsilon = 0.05; % works 
    %%% PLEASE DON'T CHANGE THIS %%%%%

    % Set the pause value for figure display:
    pause_val = 0.006;  % unit: s
    
    % Rescale the double normalized coefficient:
    normalized_coeff = pred_coeff * norm_factor;
    
    % Initialize cell arrays:
    interp_points = cell(size(pred_coeff, 2), 1);
    ind_filtered = cell(size(pred_coeff, 2), 1);
    interp_pdf = cell(size(pred_coeff, 2), 1);
    finalPdf = cell(size(pred_coeff, 2), 1);
    integ_1d = cell(size(pred_coeff, 2), 1);
    cumInteg = cell(size(pred_coeff, 2), 1);
    I_total = zeros(size(pred_coeff, 2), 1);
    
    % Provide the common string for figure 5:
    fig5_str = 'grid_filtering_';
    
    % Create a figure object to display the grid filtering output:
    f5 = figure('units','normalized','outerposition',[0 0 1 1]);

    % Provide the common string for figure 6:
    fig6_str = 'marg_pdf_';    
    
    % Create a figure object to display the PDF contours, 1D marginal PDF and
    % cumulative integral:
    f6 = figure('units','normalized','outerposition',[0 0 1 1]);

    % Loop over the predicted coefficients:
    for kk = 1:size(pred_coeff, 2)
    
        % Collect the extremal bounds for this time instant:
        extremal_bounds1 = [squeeze(integ_output{2, 1}(kk,1,:))...
            squeeze(integ_output{2, 1}(kk,2,:))];
    
        % Assemble the approximation nodes:
        approx_nodes = [squeeze(integ_output{3, 1}(kk,1,:)) squeeze(integ_output{3, 1}(kk,2,:))];
    
        % Generate the interpolating grid:
        [X, Y] = ndgrid(linspace(min(extremal_bounds1(:,1)),...
            max(extremal_bounds1(:,1)), inter_grid_res), ...
            linspace(min(extremal_bounds1(:,2)),...
            max(extremal_bounds1(:,2)), inter_grid_res));
        query_points = [X(:) Y(:)];
        
        % Perform the grid filtering:
        inPoints_interp = inhull2(query_points, extremal_bounds1);
        
        % Collect the filtered points:
        interp_points{kk, 1} = query_points(inPoints_interp,:);
        
        % Obtain the filterd indices for later use:
        ind_filtered{kk, 1} = find(inPoints_interp);

        % Plot the grid filtering for this time instant:
        figure(f5);
        subplot(1,2,1)
        plot( extremal_bounds1(:,1), extremal_bounds1(:,2),'r*',...
            query_points(:,1), query_points(:,2), 'b.')
        grid on  
        xlabel('Position (m)')
        ylabel('Velocity (m/s)')
        title(sprintf('Before filtering: 6 \\sigma bounds (red) and interpolation points (blue) (t_f = %0.2f s)',tspan(kk)))
        
        subplot(1,2,2)
        plot( extremal_bounds1(:,1), extremal_bounds1(:,2),'r*',...
            interp_points{kk, 1}(:,1), interp_points{kk, 1}(:,2), 'b.')
        grid on
        xlabel('Position (m)')
        ylabel('Velocity (m/s)')
        title(sprintf('After filtering: 6 \\sigma bounds (red) and interpolation points (blue) (t_f = %0.2f s)',tspan(kk)))        
    
        % Interpolate the PDF on to the filtered grid points:
        normalized_interp_pdf = RBFeval2d(approx_nodes, interp_points{kk, 1},...
            normalized_coeff(:, kk), RBFtype, epsilon);
        
        % Scaling the interpolated PDF:
        interp_pdf{kk, 1} = normalized_interp_pdf*max(initial_pdf);
        
        % Obtaining the PDF values onto a rectangular grid:
        finalPdf{kk, 1} = zeros(length(query_points),1);
        finalPdf{kk, 1}(ind_filtered{kk, 1}) = interp_pdf{kk, 1};
        
        % Obtain the lower dimensional marginal PDFs and the cumulative integral from the interpolated PDF:
        integ_1d{kk, 1} = squeeze(trapz(squeeze(Y(1,:)), reshape(finalPdf{kk, 1}, size(Y)), 2));
        I_total(kk, 1) = squeeze(trapz(squeeze(X(:,1)), integ_1d{kk, 1}));
        cumInteg{kk, 1} = zeros(length(integ_1d{kk, 1}),1);
        for i = 2:length(cumInteg{kk, 1})
            cumInteg{kk, 1}(i,1) = squeeze(trapz(squeeze(X(1:i,1)), integ_1d{kk, 1}(1:i)));
        end
        
        % Plot the marginal PDFs and the cumulative integral curve:
        figure(f6);
        subplot(2,2,1)
        surf(X, Y, reshape(finalPdf{kk, 1}, size(Y)))
        title(sprintf('2D Duffing oscillator: PDF (t_f = %0.2f s)',tspan(kk)))
        xlabel('Position (m)')
        ylabel('Velocity (m/s)')
        zlabel(' PDF')
        ax=gca;
        ax.FontSize = 15;
        
        subplot(2,2,2)
        contour(X, Y, reshape(finalPdf{kk, 1}, size(Y)), [160:-10:10],'ShowText','on')
        grid on
        title(sprintf('2D Duffing oscillator: contours of the PDF (t_f = %0.2f s)',tspan(kk)))
        xlabel('Position (m)')
        ylabel('Velocity (m/s)')
        ax=gca;
        ax.FontSize = 15;
        
        subplot(2,2,3)
        plot(X(:,1), integ_1d{kk, 1}, 'LineWidth', 2)
        grid on
        title(sprintf('2D Duffing oscillator: 1D marginal PDF (t_f = %0.2f s)',tspan(kk)))
        xlabel('Position (m)')
        ax=gca;
        ax.FontSize = 15;
        
        subplot(2,2,4)
        plot(X(:,1), cumInteg{kk, 1}, 'LineWidth', 2)
        grid on
        title(sprintf('2D Duffing oscillator: cumulative integral (t_f = %0.2f s)',tspan(kk)))
        xlabel('Position (m)')
        ax=gca;
        ax.FontSize = 15;

        % Save the figures if necessary:
        if isequal(figure_save, 1)
            saveas(f5, strcat(fig5_str , num2str(kk), '.png'))
            saveas(f6, strcat(fig6_str , num2str(kk), '.png'))
        end
    
        % Pause each figure in the loop for visualizing:
        pause(pause_val)
    
    end
    
    % Make movies using saved images if necessary:
    if isequal(figure_save, 1) && isequal(create_video, 1)
        create_video_from_images(size(pred_coeff, 2), fig5_str)
        create_video_from_images(size(pred_coeff, 2), fig6_str)
    end

end