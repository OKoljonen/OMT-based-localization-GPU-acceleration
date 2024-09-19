function [U, V] = sinkhorn_log_gpu(A, B, K, C, epsilon, maxIter)
    % Move data to GPU and work in log domain
    A_gpu = log(gpuArray(A));
    B_gpu = log(gpuArray(B));
    K_gpu = gpuArray(K);

    V = zeros(size(B_gpu), 'gpuArray');  % Start with log(1) = 0
    U = zeros(size(A_gpu), 'gpuArray');  % Start with log(1) = 0

    for iter = 1:maxIter
        % Update U and V in log domain
        U = A_gpu - log(K_gpu * exp(V));  % log-domain updates
        V = B_gpu - log(K_gpu' * exp(U)); % log-domain updates
    end
    
    % Convert back from log domain to standard values
    U = exp(U);
    V = exp(V);

    % Gather results back to CPU if needed
    U = gather(U);
    V = gather(V);
end
