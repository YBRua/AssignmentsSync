%% init
clear;
clc;

%% dft by def
profile off;
exec_time_dft_def = zeros(1);

i = 1;
for length_exponential = 2 : 2 : 400
    disp(length_exponential);
    current_length = 2^length_exponential;
    mat_cpu = randn(current_length, 1);
    profile on
    dft_def = DFT_by_def(mat_cpu);
    profile_data = profile('info');
    profile off;
    for j = 1 : length(profile_data.FunctionTable)
        current_profile = profile_data.FunctionTable(j);
        if strcmp(current_profile.FunctionName, 'DFT_by_def')
            exec_time_dft_def(i) = current_profile.TotalTime;
        end
    end
    i = i + 1;
end

%% dft by matmul
profile off;
exec_time_dft_mat = zeros(1);

i = 1;
for length_exponential = 2 : 2 : 400
    disp(length_exponential);
    current_length = 2^length_exponential;
    mat_cpu = randn(current_length,1);
    profile on
    dft_matmul = DFT_by_matmul(mat_cpu);
    profile_data = profile('info');
    profile off;
    for j = 1 : length(profile_data.FunctionTable)
        current_profile = profile_data.FunctionTable(j);
        if strcmp(current_profile.FunctionName, 'DFT_by_matmul')
            exec_time_dft_mat(i) = current_profile.TotalTime;
        end
    end
    i = i + 1;
end

%% fft on cpu
profile off;
exec_time_fft_cpu = zeros(1);

i = 1;
for length_exponential = 2 : 2 : 400
    disp(length_exponential);
    current_length = 2^length_exponential;
    mat_cpu = randn(current_length, 1);
    profile on;
    fft_cpu = FFT_on_CPU(mat_cpu);
    profile_data = profile('info');
    profile off;
    for j = 1 : length(profile_data.FunctionTable)
        current_profile = profile_data.FunctionTable(j);
        if strcmp(current_profile.FunctionName, 'FFT_on_CPU')
            exec_time_fft_cpu(i) = current_profile.TotalTime;
        end
    end
    i = i + 1;
end

%% fft on gpu
profile off;
exec_time_fft_gpu = zeros(1);

i = 1;
for length_exponential = 2 : 2 : 400
    disp(length_exponential);
    current_length = 2^length_exponential;
    mat_cpu = randn(current_length, 1);
    mat_gpu = gpuArray(mat_cpu);
    profile on;
    fft_cpu = FFT_on_GPU(mat_gpu);
    profile_data = profile('info');
    profile off;
    for j = 1 : length(profile_data.FunctionTable)
        current_profile = profile_data.FunctionTable(j);
        if strcmp(current_profile.FunctionName, 'FFT_on_GPU')
            exec_time_fft_gpu(i) = current_profile.TotalTime;
        end
    end
    profile off;
    i = i + 1;
end

%% visualization
profile off;
figure(1)
semilogy([2:2:2 * length(exec_time_fft_cpu)], exec_time_fft_cpu, '-o')
hold on;
semilogy([2:2:2 * length(exec_time_fft_gpu)], exec_time_fft_gpu, '-x')
semilogy([2:2:2 * length(exec_time_dft_def)], exec_time_dft_def, '-d')
semilogy([2:2:2 * length(exec_time_dft_mat)], exec_time_dft_mat, '-s')
legend('FFT on CPU', 'FFT on GPU', 'DFT by def', 'DFT by matmul')
xlabel('Input scale')
ylabel('Execution Time')
xlim([0 30])
saveas(gcf, 'result.png')
saveas(gcf, 'result.eps', 'epsc')