function [ X, Y ] = Non_IID( x, y, N, client_label)
% 用于将数据样本[x, y]按照Non_IID的方式分配给N个客户端，
% client_label表示每个客户端被分配到的类标的数量，一般取client_label=2，
% 也可以取1（训练难度大），越大难度越低。

%% 重新排序
[m, ~] = size(x);
[~, y_temp_1] = max(y, [], 2);%把类标转化为1维数据
Data = sortrows([x, y_temp_1], size(x, 2)+1);%按类标进行排序
sli_sam_num = floor(m/(client_label * N));%每个块的样本数量；

%% 给每个小块分配样本
for i=1:client_label * N
    if i == 1%这个判断是为了把头尾两块分配好
        slice_x{i} = Data(1 : i*sli_sam_num, :);
    elseif i== client_label * N
        slice_x{i} = Data((i-1)*sli_sam_num+1 : end, :);
    else
        %这个判断是为了防止某个块出来两个类标的样本
        if Data((i-1)*sli_sam_num+1, end) == Data(i*sli_sam_num, end)
            slice_x{i} = Data((i-1)*sli_sam_num+1 : i*sli_sam_num, :);
        else
            temp = find(Data(:, end)==Data((i-1)*sli_sam_num+1, end));
            temp = temp(end);
            slice_x{i} = Data((i-1)*sli_sam_num+1 : temp, :);
        end
    end
end

%% 给客户端分配数据块
x_temp = cell(1,N);
for i=1:N
    for j=1:client_label
        x_temp{i} = [x_temp{i};slice_x{(j-1)*N + i}];
    end
end

for i=1:N
    X{i} = x_temp{i}(:,1:end-1);
    y_temp_2{i} = x_temp{i}(:,end);
end

for i=1:N
    Y{i} = zeros(size(y_temp_2{i},1), size(y, 2));
end

for i=1:N
    for j=1:size(Y{i}, 1)
        Y{i}(j, y_temp_2{i}(j)) = 1;
    end
end
end