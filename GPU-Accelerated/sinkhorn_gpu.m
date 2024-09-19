function [U, V] = sinkhorn_gpu(A, B, K, C, epsilon, maxIter)
    % A, B: Histograms (TDOA measurements in matrix form)
    % K: Kernel matrix (exp(-C/epsilon)) 
    % C: Ground cost matrix
    % epsilon: Regularization parameter
    % maxIter: Maximum number of Sinkhorn iterations

    % Move data to GPU
    A_gpu = gpuArray(A); % n x N matrix
    B_gpu = gpuArray(B); % m x N matrix
    K_gpu = gpuArray(K); % Kernel matrix on the GPU

    % Initialize V and U
    V = ones(size(B_gpu), 'gpuArray');  % m x N matrix, initialized to 1
    U = ones(size(A_gpu), 'gpuArray');  % n x N matrix, initialized to 1

    for iter = 1:maxIter
        % Update U and V using Sinkhorn iterations on GPU
        U = A_gpu ./ (K_gpu * V);  % Elementwise division and matrix multiplication
        V = B_gpu ./ (K_gpu' * U); % Elementwise division and matrix multiplication
    end
    
    % Gather results back to CPU if needed
    U = gather(U);
    V = gather(V);
end
