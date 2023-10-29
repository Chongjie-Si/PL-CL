function W = obtain_W(train_data, y, k,lambda,mu)

[p,~]=size(y);
train_data = normr(train_data);
kdtree = KDTreeSearcher(train_data);
[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
W = zeros(p,p);

options = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex' );
W = zeros(p,p);

for i = 1:p
	train_data1 = train_data(neighbor(i,:),:);
	D = repmat(train_data(i,:),k,1)-train_data1;
	DD = D*D';
	y1 = y(neighbor(i,:),:);
	Dy = repmat(y(i,:),k,1)-y1;
	DyDy = Dy*Dy';
	DDDD = lambda*DD + mu*DyDy;
	lb = sparse(k,1);
	ub = ones(k,1);
	Aeq = ub';
	beq = 1;
	w = quadprog(2*DDDD, [], [],[], Aeq, beq, lb, ub,[], options);
	W(i,neighbor(i,:)) = w';
end
fprintf('\n')
end

