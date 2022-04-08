function   [NetoutTest,Training_time,Testing_time,TrainingAccuracy,TestingAccuracy]  = bls_train(train_x,train_y,test_x,test_y,Alpha,WeightEnhan,s,C,NumRule,NumFuzz);
std = 1;
tic
% train_x = zscore(train_x')';
H1 = train_x;
y=zeros(size(train_x,1),NumFuzz*NumRule);
for i=1:NumFuzz
    b1=Alpha{i};
    t_y= zeros(size(train_x,1), NumRule);
     [~,center] = kmeans(train_x, NumRule,'emptyaction','singleton');
 %   [center,~] = fcm(train_x,N11,[NaN NaN NaN false]);
    for j = 1:size(train_x,1)
        MF = exp(-(repmat(train_x(j,:), NumRule,1) - center).^2/std);
        MF = prod(MF,2);
        MF = MF/sum(MF);
        t_y(j,:) = MF'.*(train_x(j,:)*b1);
    end
%%%%%%%%%%%%%%%%%%%%% FUZZY CM%%%%%%%%%%%%%%%%%%%%%%  
%     [center,MF] = fcm(train_x,N11,[NaN NaN NaN false]);
%     t_y = MF'.*(train_x*b1);
%%%%%%%%%%%%%%%%%%%%% FUZZY CM%%%%%%%%%%%%%%%%%%%%%%  

    CENTER{i}=center;
    T1 = t_y;
    [T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';
    ps(i)=ps1;
    y(:,NumRule*(i-1)+1:NumRule*i)=T1;
 end
clear H1;
clear T1;
H2 = [y,  0.1 * ones(size(y,1),1)];


T2 = H2 * WeightEnhan;
l2 = max(max(T2));
l2 = s/l2;


T2 = tansig(T2 * l2);%tansig(x)=2/(1+exp(-2*x))-1
T3=[y T2];
clear H2;clear T2;
 beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);
Training_time = toc;
 disp('Training has been finished!');
 disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
NetoutTrain = T3 * beta;
 clear T3;
 
yy = result_tra(NetoutTrain);
train_yy = result_tra(train_y);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
tic;

% test_x = zscore(test_x')';
HH1 = test_x;
yy1=zeros(size(test_x,1),NumFuzz*NumRule);
for i=1:NumFuzz
    b1=Alpha{i};
    t_y= zeros(size(test_x,1), NumRule);
    center = CENTER{i};
    for j = 1:size(test_x,1)
        MF = exp(-(repmat(test_x(j,:), NumRule,1) - center).^2/std);
        MF = prod(MF,2);
        MF = MF/sum(MF);
        t_y(j,:) = MF'.*(test_x(j,:)*b1);
    end

%%%%%%%%%%%%%%% For FUZZY C-MEANS: COMPUTING MEMBERSHIP FUNCTION%%%%%%%%%%%%%
%     for j=1:N11
%          t_y(:,j)= sqrt(sum((test_x-repmat(center(j,:), size(test_x,1),1)).^2,2));
%     end
%     t_y=1./(t_y.^2.*(sum(t_y.^(-2),2)*ones(1,N11)));
%     t_y = t_y.*(test_x*b1);
%%%%%%%%%%%%%%% For FUZZY C-MEANS: COMPUTING MEMBERSHIP FUNCTION%%%%%%%%%%%%%
    ps1=ps(i);
    TT1 = t_y;
    TT1  =  mapminmax('apply',TT1',ps1)';
    clear beta1; clear ps1;
    yy1(:,NumRule*(i-1)+1:NumRule*i)=TT1;
    clear beta1;
    clear ps1;
end
clear TT1;clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * WeightEnhan * l2);

TT3=[yy1 TT2];
clear HH2;clear b2;clear TT2;

NetoutTest = TT3 * beta;

y = result_tra(NetoutTest);
test_yy = result_tra(test_y);
TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);
 clear TT3;
%% Calculate the testing accuracy
Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
