function results = fda_multi_convex_classify_v2(kernels, y, model)%, method)

% Project each view
k = length(kernels);
for i=1:k
    [m,n] = size(kernels{i});
    nf = length(model.alpha{i});
    if(n == nf)
        K = kernels{i};
    else
        K = kernels{i}';
    end

    P(:,i) = K*model.alpha{i} + model.b(i);
    c(:,i) = sign(P(:,i));
    e(i) = mean(c(:,i)~=y);
end

%P = [Px Py];

% if(method == 1)
%     % Empirical risk minisation principle
%     [v idx] = min(e);
%     decvals = P(:,idx);
%     yest = c(:,idx);
%     err = v;
% end
% 
% if(method == 2)
%     % Additive cost function
%     decvals = sum(P,2);
%     yest = sign(decvals);
%     err = mean(yest~=y);
% end


% Additive cost function
decvals = sum(P,2);
yest = sign(decvals);
err = mean(yest~=y);


%results.u = sum(P);              % Effective weights of kernels
results.u = sqrt(sum(P.^2))/sum(sqrt(sum(P.^2)));
results.c = yest;
results.e = err;
results.eb = balanced_errate(results.c,y);
results.f = decvals;
results.y = y;
results.roc = auc(c,y);