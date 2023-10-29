function Outputs = UpdateY(W,train_p_target,train_outputs,E,q0,mu,beta)
%Update label confidence Y

[p,q]=size(train_p_target);

options = optimoptions('quadprog',...
'Display', 'iter','Algorithm','interior-point-convex' );
tic
para_t = mu/(1+beta);

T = 2*(eye(p)-W)'*(eye(p)-W)+2/para_t*eye(p);
toc
T1 = repmat({T},1,q);
M = spblkdiag(T1{:});
lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);
II = sparse(eye(p));
A = repmat(II,1,q);
b=ones(p,1);
tr = (train_outputs+beta*(E-q0)) / (1+beta);
f = reshape(tr, p*q, 1);
Outputs= quadprog(M, -2*(1/para_t)*f, [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q);
end