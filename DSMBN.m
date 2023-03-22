%%%%%%%%%%%%% This is a demonstration of the MSBLS algorithm.%%%%%%%%%%%%%
%%%%%%%%%%%%% Each data set is followed by the corresponding optimal parameters.%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%load the dataset%%%%%%%%%%%%%%%%%%%%
clear
warning off all;
format compact;
% load data_set/norb; data_name = 'norb';N1=50;N2=10;N3=10000;
% load data_set/mnist; data_name = 'mnist';N1=10;N2=20;N3=10000;
load data_set/fashion; data_name = 'fashion';N1=50;N2=20;N3=10000;
N_client = 200
round = 10;%轮次
ind = 0;%不变
inc = 0;%不变
%%%%%%%%%%%%%%%the samples from the data are normalized and the lable data
%%%%%%%%%%%%%%%train_y and test_y are reset as N*C matrices%%%%%%%%%%%%%%

%%%%%%%%%%%%%以下4种情况任选1种%%%%%%%%%%%%%%%%%%%%
%% imbalance
[ train_X, train_Y ] = Imbalance( train_x, train_y, N_client);
[ test_X, test_Y ] = Imbalance( test_x, test_y, N_client);

%% Non-iid
% [ train_X, train_Y ] = Non_IID( train_x, train_y, N_client, 2);
% [ test_X, test_Y ] = Non_IID( test_x, test_y, N_client, 2);


%% Incremental data
% [ train_X, train_Y ] = Incremental_data( train_x, train_y, N_client, round);
% [ test_X, test_Y ] = Imbalance( test_x, test_y, N_client); ind = 1;


%% Incremental client
% [ train_X, train_Y ] = Incremental_client( train_x, train_y, N_client, round);
% [ test_X, test_Y ] = Incremental_client( test_x, test_y, N_client, round); inc = 1;


%% 找出样本数量最多的两个客户端，并编号为1,2
if (ind + inc) ==0
    for i = 1 : N_client
        sam_client(i) = size(train_X{i},1);
    end
    [~, first_max] = max(sam_client);
    sam_client(first_max) = -1;
    [~, second_max] = max(sam_client);

    temp = train_X{1};
    train_X{1} = train_X{first_max};
    train_X{first_max} = temp;
    temp = train_X{2};
    train_X{2} = train_X{second_max};
    train_X{second_max} = temp;

    temp = train_Y{1};
    train_Y{1} = train_Y{first_max};
    train_Y{first_max} = temp;
    temp = train_Y{2};
    train_Y{2} = train_Y{second_max};
    train_Y{second_max} = temp;
    clear temp first_max second_max sam_client
end


%% 归一化
if (ind + inc) ==0
    for i=1:N_client
        train_X{i} = zscore(double(train_X{i})')';
        test_X{i} = zscore(double(test_X{i})')';
        train_Y{i} = (double(train_Y{i})-1)*2+1;
        test_Y{i} = (double(test_Y{i})-1)*2+1;
    end
else
    [m1, n1] = size(train_X);
    [m2, n2] = size(test_X);
    for i=1:m1
        for j=1:n1
            if isempty(train_X{i, j})-1
                train_X{i, j} = zscore(double(train_X{i, j})')';
                train_Y{i, j} = (double(train_Y{i, j})-1)*2+1;
            end
        end
    end
    for i=1:m2
        for j=1:n2
            if isempty(test_X{i, j})-1
                test_X{i, j} = zscore(double(test_X{i, j})')';
                test_Y{i, j} = (double(test_Y{i, j})-1)*2+1;
            end
        end
    end
end
%%

%%%%%%%%%%%%%%%%%%%%This is the model of MSBLS%%%%%%
c = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
% N1=50;%feature nodes  per window，even
% N2=10;% number of windows of feature nodes
% N3=10000;% number of enhancement nodes
% train_err=zeros(1,epochs);test_err=zeros(1,epochs);
% train_time=zeros(1,epochs);test_time=zeros(1,epochs);

[N, d] = size(train_x);
n = N1*N2;
d_z = 1;
M = N_client;

result=[];

disp([newline, 'mapped features=', num2str(N1*N2), ',enhancement features=', num2str(N3)]);

if (ind + inc) == 0
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] ...
        = DSMBN_Protocol_1_train(train_X,train_Y,test_X,test_Y,s,c,N1,N2,N3,N_client);
elseif ind == 1
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] ...
        = DSMBN_Protocol_2_inc_data_train(train_X,train_Y,test_X,test_Y,s,c,N1,N2,N3,N_client,round);
elseif inc == 1
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] ...
    = DSMBN_Protocol_3_inc_client_train(train_X,train_Y,test_X,test_Y,s,c,N1,N2,N3,N_client,round);
end

train_err=TrainingAccuracy * 100;
test_err=TestingAccuracy * 100;
train_time=Training_time;
test_time=Testing_time;

% result(i,1:3)=[N1,N2,N3];result(i,4:7)=[train_err, test_err, train_time, test_time]

% result = fopen([data_name '_result_feature_' num2str(N1*N2) '_enhancement_' num2str(N3) '.txt'], 'a+');
% fprintf(result, '%.4/n', 'train_err', 'test_err', 'train_time', 'test_time');
% fclose(result);
% save ( [data_name '_result_feature_' num2str(N1*N2) '_enhancement_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
[N, d] = size(train_x);
n = N1*N2;
d_z = 1;
M = N_client;

cost = (N*d+2*(M-1)*N/M*d+M*d*d_z*n+3/2*N*d_z*n)/1024/1024
% N_first = 100/550*N;%协议2的通信成本
% cost = (N_first*d+2*(M-1)*N_first/M*d+M*d*d_z*n+3/2*N_first*d_z*n)/1024/1024%第一轮的成本
% N_other= 50/550*N;
% cost = (N_other*d*2+N_other*d_z*n*2+50*d*d_z*n)/1024/1024%后续每一轮的成本
% 
% N_first = 100/190*N;%协议3的通信成本
% cost = (N_first*d+2*(M-1)*N_first/M*d+M*d*d_z*n+3/2*N_first*d_z*n)/1024/1024%第一轮的成本
% N_other = 10/190*N;
% cost = (N_other*d*2+N_other*d_z*n*2+10*d*d_z*n)/1024/1024%后续每一轮的成本

disp([newline, 'communication cost=', num2str(cost), 'M']);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
