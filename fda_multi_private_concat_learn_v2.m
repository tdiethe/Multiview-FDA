function model = fda_multi_private_concat_learn_v2(kernels, y, kappa, slacknorm, regnorm, nweights, verbose)

tic;

if(regnorm == 1)
	%nn = 1;
	nfunc = @reg1norm;
else
	%nn = 2;
	nfunc = @reg2norm;
end

ell = length(y);
idx1 = find(y==1); idx2 = find(y==-1);
k = length(kernels);

model.description = sprintf('FDAPC (%d,%d) %d-K, kappa=%.1e', slacknorm, regnorm, k, kappa);
model.slacknorm = slacknorm;
model.regularisation = regnorm;
model.nweights = nweights;
if(verbose)
    display(model.description);
end


% Create concatenated matrix
KK = horzcat(kernels{:});

mix_param = 1; %0.1

cvx_begin
    cvx_quiet(true);
    cvx_precision('high');
    cvx_solver('sedumi');
    variable xi(ell);
    variable xip(ell*k);
    variable av(ell*k);
    variable b(1);
    variable theta(k);
    %minimize(xi'*xi + reg(av,B,k,kappa,nn));
    minimize(sfunc([xi;xip],slacknorm) + feval(nfunc,av,kappa,KK,k));
    %minimize(norm(xi,1) + reg(al,be,Kxtrn,Kytrn,kappa));
    subject to
	    G = zeros(ell,1);   
        for i=1:k
            idx = ((i-1)*ell+1):i*ell;
            G = G + ((KK(:,idx) * av(idx)) + mix_param*xip(idx));
        end
        G + repmat(b, ell, 1) == y+xi;
        %KK*(av.*repmat(theta,ell,1)) + repmat(b,ell,1) == y+xi;
        sum(xi(idx1)) == 0;
        sum(xi(idx2)) == 0;
cvx_end

% Normalise weights
for i=1:k
    ss = ((i-1)*ell)+1; ee = i*ell;
    if(nweights==1)
         model.alpha{i} = normalise(av(ss:ee),kernels{i});
    else
         model.alpha{i} = av(ss:ee);
         u(i) = norm(kernels{i}*model.alpha{i});
    end
    model.b(i) = b/k;%/2;%model.b(i) = 0.25*(model.alpha{i}'*kernels{i}*rescale)/(ellplus*ellminus);
end
model.time = toc;
model.xi = xi;
model.xip = xip;
model.u = u./sum(u);

if(verbose)
    display(sprintf('Implicit weights: %s', [num2str(model.u, ' %.2f')]));
    [cc ii] = sort(u,'descend');
    display(sprintf('Ranking of kernels: %s', [num2str(ii,' %d')]));
end

function r = sfunc(xi,slacknorm)
	%r = pow_abs(norm(xi,slacknorm),slacknorm);
	if(slacknorm == 1)
		r = norm(xi,1);
	else if(slacknorm == 2)
		r = xi'*xi;
	end
end

%  function r = regprimal1norm(av,kappa,KK,k)
%  	r = kappa*norm(av,1);
%  
%  function r = regprimal2norm(av,kappa,KK,k)
%  	r = kappa*av'*av;
%  
%  function r = regdual1norm(av,kappa,KK,k)
%  	r = kappa*norm(KK*av,1);
%  
%  function r = regdual2norm(av,kappa,KK,k)
%  	%r = kappa*av'*B*av;
%          r = 0; ell = length(av)/k;
%  	%display(sprintf('ell: %d, length(av): %d, k: %d', ell, length(av), k));
%  	for i=1:k
%  	   ss = ((i-1)*ell)+1; ee = i*ell;
%  	   avs = av(ss:ee);
%             K = KK(:,ss:ee);
%  	   %display(sprintf('ss: %d, ee: %d', ss, ee));
%  	   %t = av' * B;
%  	   %t1 = t * av;
%             %r = r + kappa*(av(ss:ee)' * B(ss:ee,ss:ee) * av(ss:ee));
%  	   r = r + kappa*(avs'*K*avs);
%         end
%  return;

%  function r = reg(av,B,k,kappa, nn)
%      if(nn == 1)
%          r = kappa*(norm(av,1));
%      elseif(nn == 2)
%          r = kappa*(av' * av);
%      elseif(nn == 4)
%          r = kappa*(av' * B * av);
%  %         r = 0; ell = length(av)/k;
%  %         for i=1:k
%  %             ss = ((i-1)*ell)+1; ee = i*ell;
%  %             r = r + kappa*(av(ss:ee)' * B(ss:ee,ss:ee) * av(ss:ee));
%  %         end
%      end
%  return;
