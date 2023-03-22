function [ X, Y ] = Imbalance( x, y, N)
% ���ڽ���������[x, y]���ղ�ƽ��ķ�ʽ�����N���ͻ��ˣ�
% �����˵����һ���ͻ��˵ķ��䵽(1/N)*(1/10)�����������������룬��ͬ��
% ����ÿ���ͻ��˵Ȳ���������������ֱ��N���ͻ���ǡ�÷���������������m��
% �����㣬����Ϊd = 2/(N-1)*(9/10)��

[m, ~] = size(x);
sample = randperm(m)';
d = 2/(N-1)*(9/10);%�Ȳ������������һ������(1/N)*(1/10)������ÿ������d��
a_1 = 1/10;
for i=1:N
    num(i) = (a_1 + (i - 1)*d)*m/N;
end
num = round(num);
num(end) = m - sum(num(1:end-1));
for i=2:N
    num(i) = num(i-1)+num(i);
end

%% ���ͻ��˷������������Ȳ����з���
X{1} = x(sample(1:num(1)), :);
for i=2:N
    X{i} = x(sample(num(i-1)+1:num(i)), :);
end
    
Y{1} = y(sample(1:num(1)), :);
for i=2:N
    Y{i} = y(sample(num(i-1)+1:num(i)), :);
end
end

