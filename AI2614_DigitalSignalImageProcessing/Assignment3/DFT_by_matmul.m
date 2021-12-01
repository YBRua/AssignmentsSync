function F = DFT_by_matmul(f)
    N = length(f);
    A = dftmtx(N);
    F = A * f;
end