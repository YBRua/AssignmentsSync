%% load
clear; clc;

img = imread('roman.jpg');
red = img(:,:,1);
green = img(:,:,2);
blue = img(:,:,3);

figure('Name', 'Original Image (R Channel)');
subplot(211);
imshow(red);
title('Image');
subplot(212);
histogram(red, 'EdgeColor', 'None', 'FaceAlpha', 0.7, 'FaceColor', '#0072BD');
title('Histogram');

%% histogram equalization
red_he = histeq(red);

figure('Name', 'Equalized Image');
% plot image
subplot(211);
imshow(red_he);
title('Image');
% plot hist
subplot(212);
histogram(red, 'EdgeColor', 'None', 'FaceAlpha', 0.5, 'FaceColor', '#4DBEEE');
hold on;
histogram(red_he, 'EdgeColor', 'None', 'FaceAlpha', 0.7, 'FaceColor', '#0072BD');
legend('Original', 'Equalized');
title('Histogram');

%% histogram specification: exponential
% create specified histogram
idx = 0 : 1 : 255;
lambda = 0.025;
exponential = lambda .* exp(-lambda .* idx);
red_exp = histeq(red, exponential);

figure('Name', 'Exponential Specification');
% plot image
subplot(211);
imshow(red_exp);
title('Image');
% plot hist
subplot(212);
histogram(red, 'EdgeColor', 'None', 'FaceAlpha', 0.5, 'FaceColor', '#4DBEEE');
hold on;
histogram(red_exp, 'EdgeColor', 'None', 'FaceAlpha', 0.7, 'FaceColor', '#0072BD');
legend('Original', 'Exponential');
title('Histogram');

%% histogram specification: gaussian
mu = 128;
sigma2 = 1600;
gaussian = 1 / (sqrt(2 * pi * sigma2)) * exp(- (idx-mu) .* (idx-mu) ./ (sigma2));
red_gaussian = histeq(red, gaussian);

figure('Name', 'Gaussian Specification');
% plot image
subplot(211);
imshow(red_gaussian);
title('Image');
% plot hist
subplot(212);
histogram(red, 'EdgeColor', 'None', 'FaceAlpha', 0.5, 'FaceColor', '#4DBEEE');
hold on;
histogram(red_gaussian, 'EdgeColor', 'None', 'FaceAlpha', 0.7, 'FaceColor', '#0072BD');
legend('Original', 'Gaussian');
title('Histogram');

%% histogram specification: sine wave
sine = sin(idx ./ 255 .* pi);
red_sine = histeq(red, sine);

figure('Name', 'Sine Wave Specification');
% plot image
subplot(211);
imshow(red_sine);
title('Image');
% plot hist
subplot(212);
histogram(red, 'EdgeColor', 'None', 'FaceAlpha', 0.5, 'FaceColor', '#4DBEEE');
hold on;
histogram(red_sine, 'EdgeColor', 'None', 'FaceAlpha', 0.7, 'FaceColor', '#0072BD');
legend('Original', 'Sine');
title('Histogram');

%% RGB equalization
green_he = histeq(green);
blue_he = histeq(blue);
img_he = img;
img_he(:,:,1) = red_he;
img_he(:,:,2) = green_he;
img_he(:,:,3) = blue_he;

figure('Name', 'RGB Equalization');
imshow(img_he);

%% CLAHE RGB equalization
% thank you, CLAHE!
img_gray = rgb2gray(img);
img_gray_he = histeq(img_gray);

img_he_impr = img;
img_he_impr(: ,: , 1) = adapthisteq(red);
img_he_impr(: ,: , 2) = adapthisteq(green);
img_he_impr(: ,: , 3) = adapthisteq(blue);

imshow(img_he_impr);

