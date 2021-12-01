function F = DFT_by_def(f)
    N = length(f);
    F = zeros(N,1);
    WN = exp(-1i*2*pi/N);
    for k = 1 : N
        x_k = 0;
        for n = 1 : N
            W_kn = WN^((k-1)*(n-1));
            x_k = x_k + f(n) * W_kn;
        end
        F(k) = x_k;
    end
end