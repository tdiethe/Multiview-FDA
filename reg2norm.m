function r = reg2norm(av,kappa,B,k)

%r = kappa*av'*av;
r = kappa*norm(av,2);
%r = kappa * av'*B*av;

return;

