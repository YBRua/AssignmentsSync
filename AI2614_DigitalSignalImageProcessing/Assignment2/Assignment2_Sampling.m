clear;
clc;
%% Parameters
start_time = -500;
end_time = 500;
sample_rate = 5;

T = 10;
amplitude = 1;
sample_time = 1/sample_rate;
t = (start_time : sample_time : end_time)'; % Construct time array
num_samples = length(t);
f = (-(num_samples-1)/2:(num_samples-1)/2)' * pi * 2 / num_samples;


%% Sampling, no delay
% sampling
x_sampled = zeros(num_samples, 1);
for i = 1 : num_samples
    x_sampled(i) = RectSignal(t(i), T/2, 0);
end

% fft
X = fft(x_sampled);
% manually reset phase
X = X .* exp(j * f * (num_samples-1)/2);
% shift to [-pi, pi]
X = fftshift(X);

% visualization
figure(1);
stem(t,x_sampled);
title('Time Domain');
xlabel('n');
ylabel('x[n]');
xlim([-10,10]);
ylim([-0.25,1.25]);

figure(2);
subplot(2,1,1);
plot(f, abs(X));
title('Amplitude');
xlabel('Frequency');
subplot(2,1,2);
plot(f, angle(X));
title('Phase');
xlabel('Frequency');

%% Sampling, delayed
% sampling
x_sampled = zeros(num_samples, 1);
for i = 1 : num_samples
    x_sampled(i) = RectSignal(t(i), T/2, sample_time/2);
end

% fft
X = fft(x_sampled);
% manually reset phase
X = X .* exp(j * f * (num_samples-1)/2);
% shift to [-pi,pi]
X = fftshift(X);

% visualization
figure(3);
stem(t,x_sampled);
title('Time Domain');
xlabel('n');
ylabel('x[n]');
xlim([-10,10]);
ylim([-0.25,1.25]);

figure(4);
subplot(2,1,1);
plot(f, abs(X));
title('Amplitude');
xlabel('Frequency');
subplot(2,1,2);
plot(f, angle(X));
title('Phase');
xlabel('Frequency');

%% Sampling, filtered, delayed.
% sampling
x_sampled = zeros(num_samples, 1);
for i = 1 : num_samples
    % sample a low-pass filtered rectangle pulse
    % pass band freq = 2Hz
    x_sampled(i) = FilteredRectSignal(t(i), T/2, 2, sample_time/2);
end

% fft
X = fft(x_sampled);
% manually reset phase
X = X .* exp(j * f * (num_samples-1)/2);
% shift to [-pi,pi]
X = fftshift(X);

% visualization
figure(5);
stem(t,abs(x_sampled));
title('Time Domain');
xlabel('n');
ylabel('x[n]');
xlim([-10,10]);
ylim([-0.25,1.25]);

figure(6);
subplot(2,1,1);
plot(f, abs(X));
title('Amplitude');
xlabel('Frequency');
subplot(2,1,2);
plot(f, angle(X));
title('Phase');
xlabel('Frequency');