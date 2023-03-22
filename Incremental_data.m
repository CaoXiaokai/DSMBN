function [ X, Y ] = Incremental_data( x, y, N, round)
% ���ڽ���������[x, y]����ƽ��ķ�ʽ�����round�ֵ�N���ͻ��ˣ�
% �����˵����һ�������пͻ��˶�����䵽���ݣ�
% �ڶ��ֿ�ʼ��ÿ����һ��ͻ��˳����������ݣ����ѡ�񣩣�
% ÿ�Ρ�ÿ���ͻ��˷��䵽��������һ����ģ�������һ�Σ���

[m, ~] = size(x);
allot_mat(1, :) = N * ones(1, N);
for i = 2 : round
    %�������Ԫ�ش���ĳ��ֵ��λ�÷������ݣ�����λ�ò���������
    allot_mat = [allot_mat; randperm(N)];
end
allot_mat(allot_mat <= N/2) = 0;
allot_mat(allot_mat > 0) = 1;
data_num = floor(m/(sum(sum(allot_mat))));

%% ��������
Data = sortrows([x, y, randperm(m)'], size(x, 2) + size(y, 2) + 1);%���ݴ���
x = Data( : , 1 : size(x, 2));
y = Data( : , size(x, 2) + 1 : size(x, 2) + size(y, 2));

%% ��ÿ���ͻ��˲�ͬ���η������ݣ�ƽ������
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

