%% The Run file for the F16 data set
clear all
close all
clc


%% Subset selections  
% p: practice / training
% t: test / validation
n=2;
d=16;

%%% Practice data
label1 = ones(0.5*(n^d),1);
label2 = -ones(0.5*(n^d),1);
labels = [label1;label2];
[Spiral1_Xp,Spiral1_Yp] = SpiralFunction(6,180,n,d,0);
[Spiral2_Xp,Spiral2_Yp] = SpiralFunction(6,180,n,d,pi);

X_p = ([Spiral1_Xp,Spiral1_Yp;Spiral2_Xp,Spiral2_Yp]);

%%% Test data                             
[Spiral1_Xt,Spiral1_Yt] = SpiralFunction(6,180,n,d,0);
[Spiral2_Xt,Spiral2_Yt] = SpiralFunction(6,180,n,d,pi);

X_t = ([Spiral1_Xt,Spiral1_Yt;Spiral2_Xt,Spiral2_Yt]);

labels_correct_validation = [ones(0.5*(n^(d)),1);-ones(0.5*(n^(d)),1)]; %already sorted!

figure(1)
plot(X_p(:,1),X_p(:,2),'g*')
hold on
plot(X_t(:,1),X_t(:,2),'k.')

%% Kernel, Hyperparameters, NARX structure selections

%%% Assign kernel parameters
KernelFunc = 'RBF';      % 'Linear' or 'RBF' 
gam  = 0.05;                                                    
sig2 = 5e-5; 
b    = 0;   

%% Early stopping

%%%%% Convergence conditions %%%%%
ConvCond.EarlyStopping = 'no';
ConvCond.FactorRemainingUncertainty         = 0.001;%   % relative to P0 = percentage - of squared sv's
ConvCond.FactorMinUncertaintChangeIter      = 0;%1/(n^n);                                   % factor/percentage uncertainty change per iteration of the squared sv's
ConvCond.FactorMinUncertaintChangeNumIter   = n^d; %round(0.01*(n^n));    % after passing through 1% of rows, if the trace change stays below...
ConvCond.SumAlphaBound                      = 0.005;  %0.0001;                 % total error of normalized ('preprocessed system')


%% Data initialization


%%% Create a LSSVM data structure
LSSVM.type = 'classification';       % 'classification' or 'regression'
LSSVM.KernelFunc = KernelFunc; 
LSSVM.Xp = X_p;
LSSVM.OutputVec = [0;ones(n^d,1)];
LSSVM.gamma = gam;
LSSVM.sig2 = sig2;
LSSVM.labels = labels;


%% Intialization of Kalman system ("KF" structure)

%%%%% Forgetting factor - covariance TT iterations %%%%%
lambda = 1;

%%%%% Create the TTV of weight vector %%%%%
scaling_m0 = 0; % only the alpha weights
m0 = TT_class.GenRankOneTT(n,d,1,scaling_m0);

%%%%% Create the initial R measurement scalar %%%%%
scaling_R =  1e-5;  
R = scaling_R;      

%%%%% Create the initial Q TTM noise - set scaling to zero if unused  %%%%%
scaling_Q = 0; 
Q0 = TT_class.GenRankOneTT(n,2*d,2,scaling_Q);

%%%%% Create the initial P TTM covariance %%%%%
scaling_P0 = 1; %sigma^2
P0 = TT_class.GenRankOneTT(n,2*d,2,scaling_P0);

%%%%% Create the TTV of 1s vector %%%%%
TTVSumVector = TT_class.GenRankOneTT(n,d,1,1);

%%%%% Rank truncation for system TT's %%%%%
%Training
Trunc_Par.DefaultMaxR           = inf; 
Trunc_Par.DefaultMaxEps         = 0; 
Trunc_Par.RankTrunc_m           = inf; 
Trunc_Par.Eps_m                 = 0.001;
Trunc_Par.RankTrunc_P           = inf;
Trunc_Par.Eps_P                 = 0.01;
Trunc_Par.RankTrunc_C           = inf;
Trunc_Par.Eps_C                 = 0.005;
Trunc_Par.RankTrunc_S_k         = inf;
Trunc_Par.Eps_S_k               = 0.005;
Trunc_Par.RankTrunc_K_k         = inf;
Trunc_Par.Eps_K_k               = 0.005;

%Validation
Trunc_Par.RankTrunc_KernelRow   = Trunc_Par.RankTrunc_C ; 
Trunc_Par.Eps_KernelRow         = Trunc_Par.Eps_C ;
Trunc_Par.MAXrank_cov_Y         = inf; 
Trunc_Par.MAXEps_cov_Y          = 0.01; 

%% Call the TTKF method to iterate over rows of the data matrix.
%%%% Explanation of function inputs:

tic
[TTKF_output, StabilityVecs] = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R,Q0,Trunc_Par,n,d,lambda,ConvCond,TTVSumVector);


%% Initialize regression with TTKF 

alpha_TT = TTKF_output(1);
covariance_TT = TTKF_output(2);  

%% Training performance  - unnecessary

[classifier_function_p,y_upperbound_p,y_lowerbound_p] = TTClassification(alpha_TT,covariance_TT,Trunc_Par,R,b,sig2,X_p,X_p,n,d,n,d,LSSVM);
toc
%%
labels_validation_mean_p= sign(classifier_function_p);
labels_validation_upperbound_p= sign(y_upperbound_p);
labels_validation_lowerbound_p= sign(y_lowerbound_p);

num_correct_p     = sum(labels_validation_mean_p== labels_correct_validation);
num_incorrect_p   = length(labels_validation_mean_p)-num_correct_p;
percentage_wrong_p= num_incorrect_p/length(labels_validation_mean_p)
percentage_right_p= num_correct_p/length(labels_validation_mean_p)

num_bounds_missclassification_p = sum((labels_validation_lowerbound_p~=labels_validation_mean_p)+(labels_validation_upperbound_p~=labels_validation_mean_p));
confidence_solution_p= 1-num_bounds_missclassification_p/(n^d)


%% figures


figure(2) % practice data
plot(Spiral1_Xp,Spiral1_Yp,'b*')
hold on
plot(Spiral2_Xp,Spiral2_Yp,'r*')


figure(3) %validation performance
idx_label1_p= labels_validation_mean_p==1;
idx_label2_p= labels_validation_mean_p==-1;
plot(X_t(idx_label1_p,1),X_t(idx_label1_p,2),'b.',X_t(idx_label2_p,1),X_t(idx_label2_p,2),'r.')
axis([-0.75 0.75 -0.75 0.75])

figure(4)
plot(classifier_function_p,'k*')
hold on
plot(y_upperbound_p,'g*')
plot(y_lowerbound_p,'g*')
grid on
title('TNKF confidence bounds - Two-sprial problem')
xlabel('instance')
ylabel('Class decision value of sorted data')
legend('Mean','Bounds')

