% GPU-accelerated version of stable_log_sum_exp
function val_out = stable_log_sum_exp_gpu(X, onesN)
    if nargin < 2
        max_X = max(X, [], 2);
        val_out = max_X + log(sum(exp(X - max_X), 2));
    else
        max_X = max(X, [], 2);
        val_out = max_X + log(sum(exp(bsxfun(@minus, X, max_X)), 2));
    end
end