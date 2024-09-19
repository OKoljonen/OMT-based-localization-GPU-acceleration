function [val_primal,val_dual,duality_gap] = eval_primal_dual_entropy_w_trash_gpu(C, trash_cost, epsilon, eta, R, lambda, mu, Psi)
    % Move input data to GPU
   

    % Perform calculations on GPU
    log_M = (-1/epsilon * C + 1/epsilon * Psi + (1/epsilon * lambda - 1/epsilon * mu'));
    M = exp(log_M);  % GPU-accelerated exponentiation
    log_m = -1/epsilon * trash_cost + 1/epsilon * lambda;
    m = exp(log_m);  % GPU-accelerated exponentiation

    % Dual value computation on GPU
    val_dual = -epsilon * (sum(M(:)) + sum(m)) - R * (R - 1) / 2 * sum(mu) + sum(lambda);

    % Primal value computation on GPU
    val_primal = C(:)' * M(:) + trash_cost(:)' * m(:) + ...
                 epsilon * (disc_ent_gpu(M, log_M) + disc_ent_gpu(m, log_m)) + ...
                 eta * sum(max(M));

    % Duality gap computation
    duality_gap = val_primal - val_dual;  % Move result back from GPU to CPU
end

% GPU-accelerated discrete entropy function
function val_out = disc_ent_gpu(M, log_M)
    val_out = M(:)' * log_M(:) - sum(M(:));  % GPU matrix multiplication and summation
end
