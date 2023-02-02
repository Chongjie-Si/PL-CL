function [train_outputs, train_outputs1, test_outputs, test_outputs1] = MulRegression_PLCL(train_data, train_p_target, q, test_data, lambda, al, par, ker)
[m, ~] = size(train_data);
[t, ~] = size(test_data);

K = kernelmatrix(ker,train_data',train_data',par);
Kt = kernelmatrix(ker,test_data',train_data',par); 

I = eye(m, m);
H = (1/(2*lambda))*K+1/2*I;
m1 = ones(m, 1);
s = (H\m1)';
P = train_p_target;
b = s*P/(s*m1);
alpha = H\(P-repmat(b, m, 1));

H1 = (al/(2*lambda)) * K + 1/2*I;
s1 = (H1\m1)';
b1 = s1*q / (s1*m1);
alpha1 = H1 \ (q-repmat(b1, m, 1));

train_outputs = 1/(2*lambda)*K*alpha+repmat(b, m, 1);
train_outputs1 = al/(2*lambda)*K*alpha1+repmat(b1, m, 1);
test_outputs = 1/(2*lambda)*Kt*alpha+repmat(b, t, 1);
test_outputs1 = al/(2*lambda)*Kt*alpha1+repmat(b1, t, 1);


end