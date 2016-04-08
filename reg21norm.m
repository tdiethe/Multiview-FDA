function r = reg2norm(av,kappa,B,k)

%r = kappa*norm(av,2);%av'*av;

%return;
% Block 2,1 norm

% Split av into constituant parts
ell = size(B,1); %length(av)/k;
for i=1:k
    ss = ((i-1)*ell)+1; ee = i*ell;
    nav(i) = norm(av(ss:ee),2);
    %nav(i) = av(ss:ee)'*av(ss:ee);
    %nav(i) = sqrt(sum(av(ss:ee).^2));
end
%r = norm(nav,1);
%r = kappa * sum(abs(nav));
r = kappa * sum(nav);

