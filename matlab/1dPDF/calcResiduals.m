clear;clc

%load truth matrix
load('asym_steeper1D_pdf_coeffs_400.mat')

%load pdf+cdf approximation
load('asym_steeper1D_pdf_cdf_400_pred.mat')

%remove cdf prediction from larger array
pred_cheb = pred_cheb(1:40,:);
pred_equi = pred_equi(1:40,:);
pred_halton = pred_halton(1:40,:);

coeff_residual_cheb_cdf = norm_cheb_rbf_coeff - pred_cheb;
coeff_residual_equi_cdf = norm_equi_rbf_coeff - pred_equi;
coeff_residual_halton_cdf = norm_halton_rbf_coeff - pred_halton;

%load just pdf prediction
load('asym_steeper1D_pdf_coeffs_400_pred.mat')

coeff_residual_cheb_pdf = norm_cheb_rbf_coeff - norm_cheb_rbf_coeff_pred;
coeff_residual_equi_pdf = norm_equi_rbf_coeff - norm_equi_rbf_coeff_pred;
coeff_residual_halton_pdf = norm_halton_rbf_coeff - norm_halton_rbf_coeff_pred;


chebPDFResidual = mean(mean(coeff_residual_cheb_pdf))
chebPDFCDFResidual = mean(mean(coeff_residual_cheb_cdf))

equiPDFResidual = mean(mean(coeff_residual_equi_pdf))
equiPDFCDFResidual = mean(mean(coeff_residual_equi_cdf))

haltonPDFResidual = mean(mean(coeff_residual_halton_pdf))
haltonPDFCDFResidual = mean(mean(coeff_residual_halton_cdf))
