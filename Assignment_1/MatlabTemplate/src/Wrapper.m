% Wrapper Code

addpath('./student_name');

clc;
clearvars;
close all;

%% Problem 1

image_path = '../Data/images/coins.png';
num_bins = 10;
[bins_vec, freq_vec, bins_vec_matlab, freq_vec_matlab] = compute_hist(image_path, num_bins);
bins_error = mean((bins_vec - bins_vec_matlab).^2);
freq_error = mean((freq_vec - freq_vec_matlab).^2);
fprintf('Error in bins: %f\n', bins_error);
fprintf('Error in freq: %f\n\n', freq_error);

%% Problem 2

gray_image_path = '../Data/images/coins.png';
[thr_w, thr_b, time_w, time_b, bin_image] = otsu_threshold(gray_image_path);
fprintf('Using within class variance: Threshold: %f, Time: %f\n', thr_w, time_w)
fprintf('Using between class variance: Threshold: %f, Time: %f\n\n', thr_b, time_b)
gray_image = imread(gray_image_path);
figure();
subplot(121); imshow(gray_image);
subplot(122); imshow(bin_image);

%% Problem 3

quote_image_path = '../Data/images/quote.png';
bg_image_path = '../Data/images/background.png';
modified_image = change_background(quote_image_path, bg_image_path);

quote_image = imread(quote_image_path);
bg_image = imread(bg_image_path);
figure();
subplot(131); imshow(quote_image);
subplot(132); imshow(bg_image);
subplot(133); imshow(modified_image);

%% Problem 4

quote_image_path = '../Data/images/quote.png';
num_characters = count_connected_components(quote_image_path);
fprintf('Number of detected letters: %d\n\n', num_characters);

%% Problem 5

noisy_image_path = '../Data/images/noisy.png';
clean_image = binary_morphology(noisy_image_path);

noisy_image = imread(noisy_image_path);
figure(); 
subplot(121); imshow(noisy_image);
subplot(122); imshow(clean_image);

%% Problem 6

gray_image_path = '../Data/images/mser.png';
mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components = count_mser_components(gray_image_path)

fprintf('Number of characters detected using MSER: %d', num_mser_components)
fprintf('Number of characters detected using Otsu: %d', num_otsu_components)
gray_image = imread(gray_image_path);
subplot(131); imshow(gray_image); title('Input Image');
subplot(132); imshow(mser_binary_image); title('MSER Binary Image'); 
subplot(133); imshow(otsu_binary_image); title('Otsu Binary Image'); 