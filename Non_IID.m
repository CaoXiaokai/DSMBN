function [ X, Y ] = Non_IID( x, y, N, client_label)
% ���ڽ���������[x, y]����Non_IID�ķ�ʽ�����N���ͻ��ˣ�
% client_label��ʾÿ���ͻ��˱����䵽������������һ��ȡclient_label=2��
% Ҳ����ȡ1��ѵ���Ѷȴ󣩣�Խ���Ѷ�Խ�͡�

%% ��������
[m, ~] = size(x);
[~, y_temp_1] = max(y, [], 2);%�����ת��Ϊ1ά����
Data = sortrows([x, y_temp_1], size(x, 2)+1);%������������
sli_sam_num = floor(m/(client_label * N));%ÿ���������������

%% ��ÿ��С���������
for i=1:client_label * N
    if i == 1%����ж���Ϊ�˰�ͷβ��������
        slice_x{i} = Data(1 : i*sli_sam_num, :);
    elseif i== client_label * N
        slice_x{i} = Data((i-1)*sli_sam_num+1 : end, :);
    else
        %����ж���Ϊ�˷�ֹĳ�������������������
        if Data((i-1)*sli_sam_num+1, end) == Data(i*sli_sam_num, end)
            slice_x{i} = Data((i-1)*sli_sam_num+1 : i*sli_sam_num, :);
        else
            temp = find(Data(:, end)==Data((i-1)*sli_sam_num+1, end));
            temp = temp(end);
            slice_x{i} = Data((i-1)*sli_sam_num+1 : temp, :);
        end
    end
end

%% ���ͻ��˷������ݿ�
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