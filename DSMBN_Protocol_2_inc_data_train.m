function [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] ...
    = DSMBN_Protocol_2_inc_data_train(train_X,train_Y,test_X,test_Y,s,c,N1,N2,N3,N_client,round)
% Learning Process of the proposed broad learning system
%Input: 
%---train_x,test_x, : the training data and learning data 
%---train_y,test_y : the label 
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N1: the number of feature nodes  per window
%----N2: the number of windows of feature nodes

%%%%%%%%%%%%%%interaction%%%%%%%%%%%%%%
for r = 1:round
    disp(['round : ', num2str(r)]);
    if r == 1
tic
for i = 1 : N_client
    sam_client(i) = size(train_X{1, i},1);
end
enh_fea_train = zeros(sum(sam_client), N1 * N2);
for k = 1 : N2
    for i = 1 : N_client
        R_i{i} = 2 * rand(size(train_X{1, i}, 1), size(train_X{1, i}, 2)) - 1;
        train_X_star{i} = train_X{1, i} + R_i{i};
    end
    for j = 1 : 2
        R_w{j} = 2 * rand(size(train_X{1, j}, 2), N1 / 2) - 1;
        temp_1 = 2 * rand(size(train_X{1, j}, 2), N1 / 2)-1;
        W{k, j} = sparse_autoencoder(train_X{1, j}, temp_1, 1e-3, 50)';
        clear temp_1
        W_star{j} = W{k, j} + R_w{j};
        
        for i = 1 : N_client
            R_ij{i, j} = 2 * rand(size(train_X{1, i}, 1), N1 / 2) - 1;
            E{i, j} = train_X_star{i} * W{k, j} + R_ij{i, j};
        end
    end
    for i = 1 : N_client
        for j = 1 : 2
            E_star{i, j} = E{i, j} - R_i{i} * W_star{j};
        end
    end
    for i = 1 : N_client
        for j = 1 : 2
            XW{i, j} = E_star{i, j} - R_ij{i, j} + R_i{i} * R_w{j};
        end
    end
    for j = 1 : 2
        XW{j, j} = train_X{1, j} * W{k, j};
    end
    C = cell2mat(XW);
    temp_2 = 2 * rand(size(C, 2), size(C, 2)) - 1;
    C_W{k} = sparse_autoencoder(C, temp_2, 1e-3, 50);
    clear temp_2
    map_fea = C * C_W{k};
    [map_fea, ps_temp]  =  mapminmax(map_fea', 0, 1);
    map_fea = map_fea';
    ps(k) = ps_temp;
    
    enh_fea_train(:, N1 * (k - 1) + 1 : N1 * k) = map_fea;
end

Encryption_time = toc;
disp(['The Total Encryption Time in Training is : ', num2str(Encryption_time), ' seconds' ]);

tic
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

enh_temp_1 = [enh_fea_train, 0.1 * ones(size(enh_fea_train, 1), 1)];
if N1 * N2 >= N3
     enh_coe = orth(2 * rand(N2 * N1 + 1, N3) - 1);
else
    enh_coe = orth(2 * rand(N2 * N1 + 1, N3)' - 1)'; 
end
enh_temp_2 = enh_temp_1 * enh_coe;
l2 = max(max(enh_temp_2));
l2 = s / l2;

enh_temp_2 = tansig(enh_temp_2 * l2);
MEfea_train=[enh_fea_train, enh_temp_2];
clear enh_temp_1;clear enh_temp_12;
Y_mat = cell2mat(train_Y(1,:)');
W_mn = (MEfea_train'  *  MEfea_train+eye(size(MEfea_train',1)) * (c)) \ ( MEfea_train'  *  Y_mat);
Enhancemen_time = toc;
Training_time = Encryption_time + Enhancemen_time; % Output training time (including encryption time)
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
xx = MEfea_train * W_mn;
clear T3;

yy = result(xx);
train_yy = result(Y_mat);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy, 1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);

    else
%% 
tic
clear sam_client enh_fea_tr XW
for i = 1 : N_client
    sam_client(i) = size(train_X{r, i},1);
end
enh_fea_train = zeros(sum(sam_client), N1 * N2);
for k = 1 : N2
    for i = 1 : N_client
        if isempty(train_X{r, i})-1
            R_i{i} = 2 * rand(size(train_X{r, i}, 1), size(train_X{r, i}, 2)) - 1;
            train_X_star{i} = train_X{r, i} + R_i{i};
        end
    end
    for j = 1 : 2
        R_w{j} = 2 * rand(size(train_X{1, j}, 2), N1 / 2) - 1;
        W_star{j} = W{k, j} + R_w{j};
        
        for i = 1 : N_client
            if isempty(train_X{r, i})-1
            R_ij{i, j} = 2 * rand(size(train_X{r, i}, 1), N1 / 2) - 1;
            E{i, j} = train_X_star{i} * W{k, j} + R_ij{i, j};
            end
        end
    end
    for i = 1 : N_client
        if isempty(train_X{r, i})-1
            for j = 1 : 2
                E_star{i, j} = E{i, j} - R_i{i} * W_star{j};
            end
        end
    end
    for i = 1 : N_client
        if isempty(train_X{r, i})-1
            for j = 1 : 2
                XW{i, j} = E_star{i, j} - R_ij{i, j} + R_i{i} * R_w{j};
            end
        end
    end
    for j = 1 : 2
        if isempty(train_X{r, j})-1
            XW{j, j} = train_X{r, j} * W{k, j};
        end
    end
    C = cell2mat(XW);
    map_fea = C * C_W{k};
    ps_temp = ps(k);
    map_fea = mapminmax('apply', map_fea', ps_temp)';
    clear ps_temp
    
%     [map_fea, ps_temp]  =  mapminmax(map_fea', 0, 1);
%     map_fea = map_fea';
%     ps(r, k) = ps_temp;
    
    enh_fea_train(:, N1 * (k - 1) + 1 : N1 * k) = map_fea;
end

    
    
Encryption_time = toc;
disp(['The Total Encryption Time in Training is : ', num2str(Encryption_time), ' seconds' ]);

tic
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

enh_temp_1 = [enh_fea_train, 0.1 * ones(size(enh_fea_train, 1), 1)];
TT2 = tansig(enh_temp_1 * enh_coe * l2);
MEfea_train=[MEfea_train; enh_fea_train, TT2];
clear enh_temp_1;clear enh_temp_12;
Y_mat = [Y_mat; cell2mat(train_Y(r,:)')];
W_mn = (MEfea_train'  *  MEfea_train+eye(size(MEfea_train',1)) * (c)) \ ( MEfea_train'  *  Y_mat);
Enhancemen_time = toc;
Training_time = Encryption_time + Enhancemen_time; % Output training time (including encryption time)
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
xx = MEfea_train * W_mn;
clear T3;

yy = result(xx);
train_yy = result(Y_mat);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy, 1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);

    end

tic;
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
if r ==1
for i = 1 : N_client
    sam_client(i) = size(test_X{i},1);
end
enh_fea_test = zeros(sum(sam_client), N1 * N2);
clear XW R RR E_1 E_2 W_star test_X_star enh_fea
for k = 1 : N2
    for i = 1 : N_client
        R_i{i} = 2 * rand(size(test_X{i}, 1), size(test_X{i}, 2)) - 1;
        test_X_star{i} = test_X{i} + R_i{i};
    end
    for j = 1 : 2
        R_w{j} = 2 * rand(size(test_X{j}, 2), N1 / 2) - 1;
        W_star{j} = W{k, j} + R_w{j};
        
        for i = 1 : N_client
            R_ij{i, j} = 2 * rand(size(test_X{i}, 1), N1 / 2) - 1;
            E{i, j} = test_X_star{i} * W{k, j} + R_ij{i, j};
        end
    end
    for i = 1 : N_client
        for j = 1 : 2
            E_star{i, j} = E{i, j} - R_i{i} * W_star{j};
        end
    end
    for i = 1 : N_client
        for j = 1 : 2
            XW{i, j} = E_star{i, j} - R_ij{i, j} + R_i{i} * R_w{j};
        end
    end
    for j = 1 : 2
        XW{j, j} = test_X{j} * W{k, j};
    end
    C = cell2mat(XW);
    
    map_fea = C * C_W{k};
    ps_temp = ps(k);
    map_fea = mapminmax('apply', map_fea', ps_temp)';
    clear ps_temp

    enh_fea_test(:, N1 * (k - 1) + 1 : N1 * k) = map_fea;
end

Encryption_time = toc;
disp(['The Total Encryption Time in Testing is : ', num2str(Encryption_time), ' seconds' ]);

tic
clear TT1;%clear HH1;
enh_temp_3 = [enh_fea_test, 0.1 * ones(size(enh_fea_test,1),1)]; 
TT2 = tansig(enh_temp_3 * enh_coe * l2);
MEfea_test=[enh_fea_test, TT2];
clear HH2;clear wh;clear TT2;
end
%%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = MEfea_test * W_mn;
yy2 = result(x);
test_yy = result(cell2mat(test_Y(1,:)'));
TestingAccuracy = length(find(yy2 == test_yy))/size(test_yy,1);
clear TT3;
Enhancemen_time = toc;
Testing_time = Encryption_time + Enhancemen_time; % Output training time (including encryption time)
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);





end

end
