function [y] = RBFeval2d(xs, x, fPar, RBFtype, R)

% Inputs:
% R - shape parameter.
% RBFtype - Type of RBF used in the original approximation.
% fPar - RBF approximation coefficients.
% xs - RBF centres in the original approximation.
% x - evaluation points.

% Including the folder path for computeDistMat():
addpath('C:\Users\pu410292\CloudStation\SynologyDrive\Drive\HigherDimensionalApproximation\RBF-PDF\vvraPaperScript')

Ns = size(xs, 1);
dim = size(xs, 2);

if size(x, 2) == dim && size(fPar, 1) == Ns
    
%     N = size(x, 1);
%     
%     r = zeros(N, Ns);
%     for i = 1:N
%         for j = 1:Ns
%             r(i, j) = norm( x(i, :) - xs(j, :) );
%         end 
%     end

    r = computeDistMat(x, xs);
    M = radialFunction(r, RBFtype, R);
    y = M*fPar;
else
    
    y = [];
    
end

end