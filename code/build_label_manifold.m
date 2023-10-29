function Outputs = build_label_manifold(train_data, train_p_target, k)
[p,q]=size(train_p_target);
train_data = normr(train_data);
kdtree = KDTreeSearcher(train_data);
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
options = optimoptions('quadprog','Display', 'off','Algorithm','interior-point-convex' );
W = zeros(p,p);
fprintf('Obtain graph matrix W...\n');
for i = 1:p
	train_data1 = train_data(neighbor(i,:),:);
	D = repmat(train_data(i,:),k,1)-train_data1;
	DD = D*D';
	lb = sparse(k,1);
	ub = ones(k,1);
	Aeq = ub';
	beq = 1;
	w = quadprog(2*DD, [], [],[], Aeq, beq, lb, ub,[], options);
	W(i,neighbor(i,:)) = w';
end
fprintf('\n')
fprintf('Generate the labeling confidence...\n');
M = sparse(p,p);
fprintf('Obtain Hessian matrix...\n');
WT = W';
T =WT*W+ W*ones(p,p)*WT.*eye(p,p)-2*WT;
T1 = repmat({T},1,q);
M = spblkdiag(T1{:});
lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);
II = sparse(eye(p));
A = repmat(II,1,q);
b=ones(p,1);
M = (M+M');
fprintf('quadprog...\n');
options = optimoptions('quadprog',...
'Display', 'iter','Algorithm','interior-point-convex' );
Outputs= quadprog(M, [], [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q);
end
