function [ X, Y ] = Incremental_client( x, y, N, round)
% 用于将数据样本[x, y]按照平衡的方式分配给N个客户端和新增客户端，
% 具体地说，第一轮中有N个客户端会分配到数据，
% 第二轮开始，每轮会新增N/10个持有数据的客户端。，
% 所有客户端持有的样本数量都是一样的。

N_new = floor(N/10) + 1;
[m, ~] = size(x);
data_num = floor(m/(N + N_new * (round - 1)));

%% 重新排序
Data = sortrows([x, y, randperm(m)'], size(x, 2) + size(y, 2) + 1);%数据打乱
x = Data( : , 1 : size(x, 2));
y = Data( : , size(x, 2) + 1 : size(x, 2) + size(y, 2));

%% 给每次新增的客户端分配数据，平均分配
num = 0;
for i = 1 : N
    X{1, i} = x(data_num * num + 1 : data_num * (num + 1), :);
    Y{1, i} = y(data_num * num + 1 : data_num * (num + 1), :);
    num = num + 1;
end

for i = 2 : round
    for j = 1 : N_new
        X{i, j} = x(data_num * num + 1 : data_num * (num + 1), :);
        Y{i, j} = y(data_num * num + 1 : data_num * (num + 1), :);
        num = num + 1;
    end
end
end


