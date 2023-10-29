clear;clc;

% load dataset (lost)
load('sample.mat');

% hyper-parameters
Maxiter = 8;
k = 10;
alpha = 0.5;
beta = 0.5;
gamma = 1;
mu = 1;
lambda = 0.03;

% please be careful
% all the data must be [number_of_samples, feature/label]
train_p_target = train_p_target';
test_target = test_target';

% parameters of kernel function
par = 1*mean(pdist(train_data));

[test_outputs, ~] = PL_CL(train_data, train_p_target, ...
    test_data, test_target, k, 'rbf', par, Maxiter, ...
    gamma, mu, lambda, alpha, beta);
