function U_out = parallel_min_logsumexp_l1_ball_gpu(V_in, eta)
    [N, M] = size(V_in);  % N is vector length, M is number of vectors

    % Sort each column in ascending order
    V = sort(V_in, 1, 'ascend');

    % Create N_cum_vec (same for all columns)
    N_cum_vec = gpuArray(N - (1:N)');

    % Check if all variables are active for each column
    sum_V = sum(V, 1);
    all_active = sum_V - N * V(1,:) < eta;

    % For columns where all variables are active
    add_const_active = sum_V(all_active) - eta;
    opt_lagrange_multiplier_active = add_const_active / N;

    % For columns where not all variables are active
    c_v = cumsum(V, 1);
    u_tilde = bsxfun(@minus, c_v(end,:), c_v) - bsxfun(@times, V, N_cum_vec);

    % Find upper_index for each column
    upper_index = sum(u_tilde >= eta, 1) + 1;
    upper_index = min(max(upper_index, 1), N);  % Ensure upper_index is within valid range

    % Calculate c, add_const, and opt_lagrange_multiplier for each column
    c = N - upper_index + 1;
    
    % Use logical indexing to handle cases where upper_index is 1
    valid_indices = upper_index > 1;
    add_const = zeros(1, M, 'gpuArray');
    add_const(valid_indices) = c_v(end, valid_indices) - c_v(sub2ind(size(c_v), upper_index(valid_indices)-1, find(valid_indices))) - eta;
    add_const(~valid_indices) = c_v(end, ~valid_indices) - eta;

    opt_lagrange_multiplier = add_const ./ c;

    % Combine results
    opt_lagrange_multiplier(all_active) = opt_lagrange_multiplier_active;

    % Calculate final output
    U_out = max(bsxfun(@minus, V_in, opt_lagrange_multiplier), 0);
end