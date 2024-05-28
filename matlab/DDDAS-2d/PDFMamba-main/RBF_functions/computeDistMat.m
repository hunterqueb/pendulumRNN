function distMat = computeDistMat(x_eval, x_center)

    if size(x_eval, 2) ~= size(x_center, 2)
        disp('Error: Dimension mismatch!')
    else
        M = size(x_eval, 1);
        N = size(x_center, 1);
        distMat = zeros(M, N);
        parfor i = 1:M
              distMat(i, :) = [vecnorm( x_eval(i, :) - x_center, 2, 2)]';
        end
    end
end


