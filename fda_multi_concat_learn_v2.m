function model = fda_multi_concat_learn_v2(kernels, y, kappa, slacknorm, regnorm, nweights, verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multiview Fisher Discriminant Analysis (concat version)
% 
% INPUTS:
%   kernels      - cell array of kernels
%   y            - labels
%   kappa        - regularisation parameter
%   slacknorm    - norm on slack variables (1 or 2)
%   regnorm      - norm on regulariser (1 or 2)
%   nweights     - normalise weights (0 = no, 1 = yes)
%   versbose     - print output (0 = no, 1 = yes)
%
% OUTPUTS:
%   model        - structure containing learnt model
%
% Not for commercial use
% Author: Tom Diethe, Department of Computer Science, UCL
%
% 30/10/2009: Slightly altered calculation of b
% 18/8/2009: First release
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

ell = length(y);
idx1 = find(y==1); idx2 = find(y==-1);
ellplus = length(idx1);
ellminus = length(idx2);
%rescale = ones(ell,1)+y*((ellminus-ellplus)/ell);

t = zeros(ell,1);
t(idx1) = 1/ellplus;
t(idx2) = 1/ellminus;

if(regnorm == 1)
	nfunc = @reg1norm;
elseif(regnorm == 2)
	nfunc = @reg2norm;
elseif(regnorm == 3)
    nfunc = @reg21norm;
elseif(isinf(regnorm))
    nfunc = @reginfnorm;
end

k = length(kernels);

model.description = sprintf('FDA C2(%d,%d) %d-K, kappa=%.1e', slacknorm, regnorm, k, kappa);
model.slacknorm = slacknorm;
model.regularisation = regnorm;
model.nweights = nweights;
if(verbose)
    display(model.description);
end

% Create concatenated matrix
KK = horzcat(kernels{:});

%tol = 1e-9;

cvx_begin
    cvx_quiet(true);
    cvx_precision('low');
    %cvx_solver('sedumi');
    
    variable xi(ell);
    variable av(ell*k);
    variable b(1);
    %variable b(k);    
    variable theta(k);
    
    minimize(sfunc(xi,slacknorm) + feval(nfunc,av,kappa,KK,k));
    
    subject to
        KK*av + repmat(b,ell,1) == y+xi;
        %(KK*av + b) == y+xi;
        sum(xi(idx1)) == 0; %<= tol;
        sum(xi(idx2)) == 0; %<= tol;
%        if(regnorm == 3)
%            for i=1:k
%                ss = ((i-1)*ell)+1; ee = i*ell;
%                nav(i) = norm(av(ss:ee),2);
%            end
%            r = sum(abs(nav));
%            r <= 1/kappa;
%        end
cvx_end

% Normalise weights (optional)
for i=1:k
    ss = ((i-1)*ell)+1; ee = i*ell;
    if(nweights==1)
         model.alpha{i} = normalise(av(ss:ee),kernels{i});
    else
         model.alpha{i} = av(ss:ee);
         u(i) = norm(kernels{i}*model.alpha{i});
    end
    %model.b(i) = b/k;
    model.b(i) = -0.5*model.alpha{i}'*kernels{i}*t;
    %model.b(i) = -0.25*(model.alpha{i}'*kernels{i}*rescale)/(ellplus*ellminus);
    %fprintf('b/k = %.2f, b = %.2f\n', b/k, model.b);
end
model.time = toc;
model.xi = xi;
model.u = u./sum(u);


if(verbose)
    display(sprintf('Implicit weights: %s', [num2str(model.u, ' %.2f')]));
    [cc ii] = sort(u,'descend');
    display(sprintf('Ranking of kernels: %s', [num2str(ii,' %d')]));
    fprintf('Execution time %.2f secs\n', model.time);
end

function r = sfunc(xi,slacknorm)
	if(slacknorm == 1)
		r = norm(xi,1);
	elseif(slacknorm == 2)
   		r = xi'*xi;
    elseif(isinf(slacknorm))
        r = norm(xi,inf);
    elseif(slacknorm == 100)
        % epsilon insensitive loss
        epsilon = 0.1;
        r = 1;
    elseif(slacknorm == 200)
        % Huber's robust loss
        sigma = 0.1;
        % Slow way for now ...
        for i=1:length(xi)
            if(abs(xi(i)) < 0.1)
                r(i) = (xi(i).^2)/(2*sigma);
            else
                r(i) = abs(sigma)/(sigma/2);
            end
        end
        %in = find(abs(xi) < 0.1);
        %r(in) = (xi(in).^2)/(2*sigma);
        %r(~in) = abs(xi(~in))-(sigma/2);
    end    


