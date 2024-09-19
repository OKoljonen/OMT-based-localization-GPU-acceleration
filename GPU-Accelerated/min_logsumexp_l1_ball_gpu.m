% GPU-accelerated version of min_logsumexp_l1_ball
function u_out = min_logsumexp_l1_ball_gpu(v_in, eta, N_cum_vec)
    v = sort(v_in, 'ascend');
    N = length(v);

    if nargin < 3
        N_cum_vec = gpuArray(N - (1:N)');
    end

    % Check if all variables are active
    if sum(v) - N * v(1) < eta
        add_const = sum(v) - eta;
        c = N;
        opt_lagrange_multiplier = add_const / c;
    else
        % Assume at least one variable is zero in optimum
        c_v = cumsum(v);
        u_tilde = c_v(end) - c_v - v.* N_cum_vec;

        upper_index = find(u_tilde < eta, 1, 'first');

        % This is the number of non-zero variables and also slope of function in this interval
        c = N - upper_index + 1;

        % Offset in this interval
        add_const = (c_v(end) - c_v(upper_index-1)) - eta;

        % Find zero of linear function. This is the Lagrange multiplier
        opt_lagrange_multiplier = add_const / c;
    end

    u_out = max(v_in - opt_lagrange_multiplier, 0);
end