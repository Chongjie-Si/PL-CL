clear;clc;
%%{
load('soccer_player.mat');
load("indices_soccerplayer.mat")
%}
Maxiter = 8;
k = 10;
alpha = 0.5;
beta = 0.5;
gamma = 1;
mu = 1;
lambda = 0.03;
  
partial_target = partial_target';
target = target';
acc=[];
for i = 1:10
    %{
    [sdf,~] = size(data);
    
    num = round(sdf/2);
    seed = round(num/10)*i+1;
    train_data = data(seed:min(seed+num,sdf),:);
    train_p_target = partial_target(seed:min(seed+num,sdf),:);
    train_target = target(seed:min(seed+num,sdf),:);
    test_target = [target(1:seed,:); target(seed+num+1:sdf,:)];
    test_data = [data(1:seed,:); data(seed+num+1:sdf,:)];
    %}
    test=(indices(:,i)==mod(i,2)+1);
    train=~test;
    train_data=data(train,:);
    test_data=data(test,:);
    test_target=target(test,:);
    train_p_target=partial_target(train,:);
    train_target=target(train,:);
    par = 1*mean(pdist(train_data)); %Parameters of kernel function

[test_outputs, ~] = PL_CL(train_data,train_p_target,test_data,test_target,k,'rbf',par,Maxiter,gamma,mu,lambda,alpha,beta);
accuracy = CalAccuracy(test_outputs, test_target);
fprintf('The accuracy of PL-CL is: %f \n',accuracy);
acc = [acc;accuracy];
end