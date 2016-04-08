function r = reginfnorm(av,kappa,B,k)

%r = kappa*av'*av;
r = kappa*norm(av,inf);
%r = kappa * av'*B*av;

return;

