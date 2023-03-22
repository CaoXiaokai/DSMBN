function [ X, Y ] = Incremental_data( x, y, N, round)
% 用于将数据样本[x, y]按照平衡的方式分配给round轮的N个客户端，
% 具体地说，第一轮中所有客户端都会分配到数据，
% 第二轮开始，每轮有一半客户端持有新增数据（随机选择），
% 每次、每个客户端分配到的数据是一样多的（包括第一次）。

[m, ~] = size(x);
allot_mat(1, :) = N * ones(1, N);
for i = 2 : round
    %分配矩阵，元素大于某个值的位置分配数据，其余位置不分配数据
    allot_mat = [allot_mat; randperm(N)];
end
allot_mat(allot_mat <= N/2) = 0;
allot_mat(allot_mat > 0) = 1;
data_num = floor(m/(sum(sum(allot_mat))));

%% 重新排序
Data = sortrows([x, y, randperm(m)'], size(x, 2) + size(y, 2) + 1);%数据打乱
x = Data( : , 1 : size(x, 2));
y = Data( : , size(x, 2) + 1 : size(x, 2) + size(y, 2));

%% 给每个客户端不同批次分配数据，平均分配
num = 0;
for i = 1 : round
    for j = 1 : N
        if allot_mat(i, j) ~= 0
            X{i, j} = x(data_num * num + 1 : data_num * (num + 1), :);
            Y{i, j} = y(data_num * num + 1 : data_num * (num + 1), :);
            num = num + 1;
        end
    end
end
end

