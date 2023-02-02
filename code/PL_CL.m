function [test_outputs, accuracy_test] = PL_CL(train_data,train_p_target,test_data,test_target,k,ker,par,Maxiter,lambda,mu,gama,al,beta)

[aaaa,bbbb] = size(train_p_target);
%y=train_p_target;
y= build_label_manifold(train_data, train_p_target, k);
q=1-train_p_target;
E = ones(aaaa,bbbb);

[train_outputs, train_outputs1, test_outputs, test_outputs1] = MulRegression_PLCL(train_data, y, q, test_data, gama, al, par, ker);
for j = 1:Maxiter
	fprintf('The %d-th iteration\n',j);
	W = obtain_W_PLCL(train_data,y,k,lambda,mu);
    q = min(1,max(1-train_p_target, (al*train_outputs1 + beta*(E-y))/(al + beta)));
	y = UpdateY_PLCL(W,train_p_target,train_outputs,E,q,mu,beta);
	[train_outputs, train_outputs1, test_outputs,test_outputs1] = MulRegression_PLCL(train_data, y, q, test_data, gama, al,par, ker);
    accuracy_test = CalAccuracy(1-test_outputs1, test_target);
    fprintf('The accuracy of PL-CL is: %f \n',accuracy_test);

end

end