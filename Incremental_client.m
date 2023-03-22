function [ X, Y ] = Incremental_client( x, y, N, round)
% ���ڽ���������[x, y]����ƽ��ķ�ʽ�����N���ͻ��˺������ͻ��ˣ�
% �����˵����һ������N���ͻ��˻���䵽���ݣ�
% �ڶ��ֿ�ʼ��ÿ�ֻ�����N/10���������ݵĿͻ��ˡ���
% ���пͻ��˳��е�������������һ���ġ�

N_new = floor(N/10) + 1;
[m, ~] = size(x);
data_num = floor(m/(N + N_new * (round - 1)));

%% ��������
Data = sortrows([x, y, randperm(m)'], size(x, 2) + size(y, 2) + 1);%���ݴ���
x = Data( : , 1 : size(x, 2));
y = Data( : , size(x, 2) + 1 : size(x, 2) + size(y, 2));

%% ��ÿ�������Ŀͻ��˷������ݣ�ƽ������
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


