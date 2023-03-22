function wk = sparse_autoencoder(X,W,lam,itrs)
% 稀疏自编码器，目标是计算稀疏的Z = XW
% 输入是归一化的数据X和系数W，输出是满足上述条件的W*
A = X * W;
A = mapminmax(A);
AA = (A') * A;
m = size(A,2);
n = size(X,2);
x = zeros(m,n);
wk = x; 
ok=x;uk=x;
L1=eye(m)/(AA+eye(m));
L2=L1*A'*X;

for i = 1:itrs,
    tempc=ok-uk;
  ck =  L2+L1*tempc;
 ok=shrinkage(ck+uk, lam);
 uk=uk+(ck-ok);
 wk=ok;
end
end
function z = shrinkage(x, kappa)
    z = max( x - kappa,0 ) - max( -x - kappa ,0);
end
% function p = objective(A, b, lam, x, z)
%     p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
% end
% % toc

