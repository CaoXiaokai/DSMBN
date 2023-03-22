function [ X, Y ] = Imbalance( x, y, N)
% 用于将数据样本[x, y]按照不平衡的方式分配给N个客户端，
% 具体地说，第一个客户端的分配到(1/N)*(1/10)的样本数（四舍五入，下同）
% 后续每个客户端等差增加样本数量，直到N个客户端恰好分配所有数据样本m，
% 经计算，公差为d = 2/(N-1)*(9/10)；

[m, ~] = size(x);
sample = randperm(m)';
d = 2/(N-1)*(9/10);%等差分配样本，第一个分配(1/N)*(1/10)，后续每个增加d。
a_1 = 1/10;
for i=1:N
    num(i) = (a_1 + (i - 1)*d)*m/N;
end
num = round(num);
num(end) = m - sum(num(1:end-1));
for i=2:N
    num(i) = num(i-1)+num(i);
end

%% 给客户端分配样本，按等差数列分配
X{1} = x(sample(1:num(1)), :);
for i=2:N
    X{i} = x(sample(num(i-1)+1:num(i)), :);
end
    
Y{1} = y(sample(1:num(1)), :);
for i=2:N
    Y{i} = y(sample(num(i-1)+1:num(i)), :);
end
end

