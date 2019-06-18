clc;
clear all;
close all;

H = [923, 8532, 249, 80, 51, 30, 22, 20, 15, 5, 7, 3, 5, 1, 0, 0, 0, 0, 0, 0, 1];
x = 0:0.05:1
H = H./sum(H);

figure;
bar(x, H)
xlabel('Difference Ratio')
ylabel('Percentage of total images')
