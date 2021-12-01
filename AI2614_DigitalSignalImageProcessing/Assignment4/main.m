%%
clear;
clc;

file_name = "HAG-Shinpakusuu#0822.mp3";
[music_origin, fs_origin] = audioread(file_name);
window_size = 0.02; % 20ms window

%% STFT of original music
window_length = window_size * fs_origin;
figure(1);
plot(music_origin, 'LineWidth', 0.1);
title('Original Music');
xlabel('Samples');
ylabel('Values');
set(gcf, 'Position', [300 300 1200 300]);
saveas(gcf, 'original_temporal.eps', 'epsc');

stft(music_origin, 'Window', hann(window_length, 'periodic'), 'FFTLength', 1024);
colormap('jet');
saveas(gcf, 'original_stft.eps', 'epsc');

%% Downsampling and Interpolating
frequencies = [5000, 10000, 15000];
interp_factor = [9, 5, 3];
for i = 1 : length(frequencies)
    % downsampling
    window_length = window_size * frequencies(i);
    music_ds = resample(music_origin, frequencies(i), fs_origin);
    music_ds = music_ds / max(abs(music_ds)); % normalize
    figure(i+1);
    plot(music_ds, 'LineWidth', 0.1);
    title(strcat(string(frequencies(i)/1000), 'kHz-', 'Downsampled'));
    xlabel('Samples');
    ylabel('Values');
    set(gcf, 'Position', [300 300 1200 300]);
    saveas(gcf, strcat(string(frequencies(i)/1000),'kHz_ds.eps'), 'epsc');

    stft(music_ds, 'Window', hann(window_length, 'periodic'), 'FFTLength', 1024);
    colormap('jet');
    saveas(gcf, strcat(string(frequencies(i)/1000),'kHz_ds_stft.eps'), 'epsc');
    
    audiowrite(strcat(string(frequencies(i)/1000), 'kHz_ds.wav'), music_ds, frequencies(i));
    
    % interpolating
    window_length = window_size * frequencies(i) * interp_factor(i);
    music_interp = interp(music_ds, interp_factor(i));
    music_interp = music_interp / max(abs(music_interp)); % normalize
    figure(i+4);
    plot(music_interp, 'LineWidth', 0.1);
    title(strcat(string(frequencies(i)/1000), 'kHz-', 'Interpolated'));
    xlabel('Samples');
    ylabel('Values');
    set(gcf, 'Position', [300 300 1200 300]);
    saveas(gcf, strcat(string(frequencies(i)/1000),'kHz_interp.eps'), 'epsc');

    stft(music_interp, 'Window', hann(window_length, 'periodic'), 'FFTLength', 1024);
    colormap('jet');
    saveas(gcf, strcat(string(frequencies(i)/1000),'kHz_interp_stft.eps'), 'epsc');
    
    audiowrite(strcat(string(frequencies(i)/1000), 'kHz_interp.wav'), music_interp, frequencies(i) * interp_factor(i));
end

%%
window_length = window_size * fs_origin;
[s,f,t] = stft(music_origin, fs_origin, 'Window', hann(window_length, 'periodic'), 'FFTLength', 1024);

y = [-5 -4 -2 0 6 7 6 4 4 2];
x = [31 63 125 250 500 1000 2000 4000 8000 16000];
y = 10 .^ (y / 20);

filter = interp1(x, y, abs(f), 'spline');

eq = filter .* s;

music_eq = istft(eq, fs_origin, 'Window', hann(window_length, 'periodic'), 'FFTLength', 1024);
music_eq = music_eq / max(abs(music_eq));
audiowrite('processed_music.wav', music_eq, fs_origin);
