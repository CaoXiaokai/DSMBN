function [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] ...
    = DSMBN_Protocol_1_train(train_X,train_Y,test_X,test_Y,s,c,N1,N2,N3,N_client)
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
tic
% A_train_x = [A_train_x, 0.1 * ones(size(A_train_x,1),1)];
% B_train_x = [B_train_x, 0.1 * ones(size(B_train_x,1),1)];
% enh_fea=zeros(size(A_train_x,1) + size(B_train_x,1),N2*N1);
for i = 1 : N_client
    sam_client(i) = size(train_X{i},1);
end
enh_fea_tr = zeros(sum(sam_client), N1 * N2);
% N1_temp = N1/2;
for k = 1 : N2
    for i = 1 : N_client
        R_i{i} = 2 * rand(size(train_X{i}, 1), size(train_X{i}, 2)) - 1;
        train_X_star{i} = train_X{i} + R_i{i};
    end
    for j = 1 : 2
        R_w{j} = 2 * rand(size(train_X{j}, 2), N1 / 2) - 1;
        temp_1 = 2 * rand(size(train_X{j}, 2), N1 / 2)-1;
        W{k, j} = sparse_autoencoder(train_X{j}, temp_1, 1e-3, 50)';
        clear temp_1
%         W{k, i} = 2 * rand(size(train_X{i}, 2), N1 / N_client)-1;
        W_star{j} = W{k, j} + R_w{j};
        
        for i = 1 : N_client
            R_ij{i, j} = 2 * rand(size(train_X{i}, 1), N1 / 2) - 1;
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
        XW{j, j} = train_X{j} * W{k, j};
    end
    C = cell2mat(XW);
    temp_2 = 2 * rand(size(C, 2), size(C, 2)) - 1;
    C_W{k} = sparse_autoencoder(C, temp_2, 1e-3, 50);
    clear temp_2
    map_fea = C * C_W{k};
    [map_fea, ps_temp]  =  mapminmax(map_fea', 0, 1);
    map_fea = map_fea';
    ps(k) = ps_temp;
    
    enh_fea_tr(:, N1 * (k - 1) + 1 : N1 * k) = map_fea;
end

Encryption_time = toc;
disp(['The Total Encryption Time in Training is : ', num2str(Encryption_time), ' seconds' ]);
% Training_Communication_Cost = sum([whos('C_RA_1','C_RB_1','C_Rb_1',...
%     'A_train_x_star','B_W_star','B_E_1','A_E_2','C_RB_2','C_RA_2','C_Ra_2',...
%     'B_train_x_star','A_W_star','A_E_1','B_E_2','XAWA','XBWB').bytes])/1024/1024;
% disp(['The Total Communication Cost in Training is : ', num2str(Training_Communication_Cost), ' MB' ]);
% [whos('A_train_x','B_train_x','y').bytes]/1024/1024


tic
% clear H1;
% clear T1;
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

enh_temp_1 = [enh_fea_tr, 0.1 * ones(size(enh_fea_tr, 1), 1)];
if N1 * N2 >= N3
     enh_coe = orth(2 * rand(N2 * N1 + 1, N3) - 1);
else
    enh_coe = orth(2 * rand(N2 * N1 + 1, N3)' - 1)'; 
end
enh_temp_2 = enh_temp_1 * enh_coe;
l2 = max(max(enh_temp_2));
l2 = s / l2;
% fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));

enh_temp_2 = tansig(enh_temp_2 * l2);
T3=[enh_fea_tr, enh_temp_2];
clear enh_temp_1;clear enh_temp_12;
W_mn = (T3'  *  T3+eye(size(T3',1)) * (c)) \ ( T3'  *  cell2mat(train_Y'));
Enhancemen_time = toc;
Training_time = Encryption_time + Enhancemen_time; % Output training time (including encryption time)
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
xx = T3 * W_mn;
clear T3;

yy = result(xx);
train_yy = result(cell2mat(train_Y'));
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy, 1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);

tic;
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
for i = 1 : N_client
    sam_client(i) = size(test_X{i},1);
end
enh_fea_te = zeros(sum(sam_client), N1 * N2);
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

    enh_fea_te(:, N1 * (k - 1) + 1 : N1 * k) = map_fea;
end

Encryption_time = toc;
disp(['The Total Encryption Time in Testing is : ', num2str(Encryption_time), ' seconds' ]);
% Testing_Communication_Cost = sum([whos('C_RA_1','C_RB_1','C_Rb_1',...
%     'A_test_x_star','B_W_star','B_E_1','A_E_2','C_RB_2','C_RA_2','C_Ra_2',...
%     'B_test_x_star','A_W_star','A_E_1','B_E_2','C_11','C_22').bytes])/1024/1024;
% disp(['The Total Communication Cost in Testing is : ', num2str(Testing_Communication_Cost), ' MB' ]);
% sum([whos('A_test_x','B_test_x','yy1').bytes])/1024/1024

tic
clear TT1;%clear HH1;
enh_temp_3 = [enh_fea_te, 0.1 * ones(size(enh_fea_te,1),1)]; 
TT2 = tansig(enh_temp_3 * enh_coe * l2);
TT3=[enh_fea_te, TT2];
clear HH2;clear wh;clear TT2;
%%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = TT3 * W_mn;
yy2 = result(x);
test_yy = result(cell2mat(test_Y'));
TestingAccuracy = length(find(yy2 == test_yy))/size(test_yy,1);
clear TT3;
Enhancemen_time = toc;
Testing_time = Encryption_time + Enhancemen_time; % Output training time (including encryption time)
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
