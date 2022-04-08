clear;
warning off all;
format compact;
load wbc  %for this data set NumRule= 2  NumFuzz=6  NumEnhan=20;

%%%%%%%%%%%% Test set For wbc data set %%%%%%%%%%%%%%%
n=randperm(size(test_x,1));
test_x = double(test_x(n(1:140),:));
test_y=double(test_y(n(1:140),:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_y=(train_y-1)*2+1;
test_y=(test_y-1)*2+1;

C = 2^-30;   %----C: the regularization parameter for sparse regualarization
s = .8;          %----s: the shrinkage parameter for enhancement nodes
best = 0.72;
result = [];
for NumRule=2 %1:1:20                 %searching range for fuzzy rules  per fuzzy subsystem
    for NumFuzz=6%1:1:20              %searching range for number of fuzzy subsystems
        for NumEnhan=20 %1:1:20    %searching range for enhancement nodes
            clc;
            rand('state',1)
            for i=1:NumFuzz
                alpha=rand(size(train_x,2),NumRule);
                Alpha{i}=alpha;
            end  %generating coefficients of the then part of fuzzy rules for each fuzzy system
            
            WeightEnhan=rand(NumFuzz*NumRule+1,NumEnhan); %%Iinitializing  weights connecting fuzzy subsystems  with enhancement layer
            
            fprintf(1, 'Fuzzy rule No.= %d, Fuzzy system. No. =%d, Enhan. No. = %d\n', NumRule, NumFuzz,NumEnhan);
            [NetoutTest,Training_time,Testing_time,TrainingAccuracy,TestingAccuracy]  = bls_train(train_x,train_y,test_x,test_y,Alpha,WeightEnhan,s,C,NumRule,NumFuzz);
            time =Training_time + Testing_time;
            result = [result; NumRule NumFuzz NumEnhan TrainingAccuracy TestingAccuracy];
            if best < TestingAccuracy
                best = TestingAccuracy;
                save optimal.mat TrainingAccuracy TestingAccuracy  NumRule NumFuzz NumEnhan time
            end
            clearvars -except best NumRule NumFuzz NumEnhan train_x train_y test_x test_y  s C result NetoutTest
        end
    end
end
save result result



