function [test_outputs, accuracy_test] = PL_CL(train_data,train_p_target,test_data,test_target,k,ker,par,Maxiter,lambda,mu,gama,al,beta)

[nt, nl] = size(train_p_target);

% labeling confidence
y = build_label_manifold(train_data, train_p_target, k);
% complementary-labeling confidence
q = 1 - train_p_target;

E = ones(nt, nl);

[train_outputs, train_outputs_com, test_outputs, ~] = MulRegression(train_data, y, q, test_data, gama, al, par, ker);
for j = 1:Maxiter
	fprintf('The %d-th iteration\n',j);

    % update W
	W = obtain_W(train_data, y, k, lambda, mu);
    % update q
    q = min(1,max(1 - train_p_target, (al * train_outputs_com + beta * (E - y)) / (al + beta)));
    % update y
	y = UpdateY(W, train_p_target, train_outputs, E, q, mu, beta);
	
    [train_outputs, train_outputs_com, test_outputs, ~] = MulRegression(train_data, y, q, test_data, gama, al, par, ker);
    accuracy_test = CalAccuracy(test_outputs, test_target);
    fprintf('The accuracy of PL-CL is: %f \n', accuracy_test);

end

end