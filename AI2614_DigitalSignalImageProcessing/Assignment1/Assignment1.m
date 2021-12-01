% AI2614 Digital Signal and Image Processing
% Programming Assignment 1: Noise and Weiner Filtering

%% Initializing PSF Matrix
clear;
clc;
close all;
filter_size = 5;
psf = fspecial('average', filter_size);

%% Loading image.
path = "./baboon.bmp";
img = im2double(imread(path));
img_max = max(img(:));

% Display
figure('Name','Original Image');
fig_origin = imshow(img);

%% Blurring
% Convolution
img_blurred = imfilter(img, psf, 'conv', 'circular');

% Display
fig_blurred = figure('Name', 'Blurred Image');
imshow(img_blurred);
imwrite(img_blurred, 'baboon_blurred.bmp');

% Update size
width = size(img_blurred, 1);
height = size(img_blurred, 2);

%% Adding Gaussian noise
snr_list = [10, 20, 30];
img_noise = zeros(width, height, length(snr_list));

for i = 1 : length(snr_list)
    % Add noise
    img_noise(:,:,i) = awgn(img_blurred, snr_list(i));
    
    % Display
    fig_noise = figure('Name', 'With Noise: ' + string(snr_list(i)) + 'dB');
    imshow(img_noise(:,:,i));
    imwrite(img_noise(:,:,i), 'baboon_noise_' + string(snr_list(i)) + '.bmp');
end

% %% Wiener filter test
% imshow(deconvwnr(img_blurred, psf));

%% Wiener Filtering
img_restored = zeros(width, height, length(snr_list));

for i = 1 : length(snr_list)
    % Calculate NSR
    current = img_noise(:,:,i);
    var_noise = var(current(:) - img(:));
    var_img = var(img(:));
    nsr = var_noise / var_img;
    
    % Restore
    current = deconvwnr(img_noise(:,:,i), psf, nsr);
    current = current ./ max(current(:)) * img_max;
    img_restored(:,:,i) = current;
    
    % Display
    fig_restored = figure('Name', 'Restored From: ' + string(snr_list(i)) + 'dB');
    imshow(img_restored(:,:,i));
    imwrite(img_restored(:,:,i), 'baboon_restored_' + string(snr_list(i)) + '.bmp');
end