%% The Run file for the F16 data set
clear all
close all
clc

%% Subset selections  
% p: practice / training
% t: test / validation


run('InitializeAdultData.m')
load('randperm_nonchanging.mat')

X_unsorted = X(randperm_nonchanging,:);              % as to mix up the data first
Labels_unsorted = Labels(randperm_nonchanging,:);    % as to mix up the data first

n_p = 3;
d_p = 9;

n_t = 3;
d_t = 7;


X_p = X_unsorted(1:n_p^d_p,:);
Labels_p = Labels_unsorted(1:n_p^d_p,:);

X_t = X_unsorted((n_p^d_p)+1:(n_p^d_p)+(n_t^d_t)+1,:); 
Labels_t = Labels_unsorted((n_p^d_p)+1:(n_p^d_p)+(n_t^d_t)+1);


%% Kernel, Hyperparameters, NARX structure selections

%%% Specify the input and output data for validation and test sets

%%% Assign kernel parameters
KernelFunc = 'RBF';     % 'linear' or 'rbf' 
gam  = 0.0015;                                           
sig2 = 0.5;  
b    = 0;   


%% Early stopping

%%%%% Convergence conditions %%%%%
ConvCond.EarlyStopping = 'yes';             % 'yes' or 'no'
ConvCond.type = 'classification';           % 'classification' or 'regression' 
ConvCond.FactorRemainingUncertainty         = 1e-10;%   % relative to P0 = percentage - of squared sv's
ConvCond.FactorMinUncertaintChangeIter      = 5e-3;    % factor/percentage uncertainty change per iteration of the squared sv's  (choose larger than 1/(n_p^d_p))
ConvCond.FactorMinUncertaintChangeNumIter   = 2; %round(0.01*(n^n));    % after passing through 1% of rows, if the trace change stays below...
ConvCond.SumAlphaBound                      = 1;  %0.0001;                 % total error of normalized ('preprocessed system')


%% Data initialization


%%% Create a LSSVM data structure
LSSVM.type = 'classification';       % 'classification' or 'regression'
LSSVM.KernelFunc = KernelFunc; 
LSSVM.Xp = X_p;
LSSVM.OutputVec = [0;ones(n_p^d_p,1)];
LSSVM.gamma = gam;
LSSVM.sig2 = sig2;
LSSVM.labels = Labels_p;


%% Intialization of Kalman system ("KF" structure)
% - Create and set parameters -- might have to change the approach. Now
% I first create matrices -> in the future create TT's directly.

%%%%% Forgetting factor - covariance TT iterations %%%%%
lambda = 0.9975; %1

%%%%% Create the TTV of weight vector %%%%%
scaling_m0 = 0; % only the alpha weights
m0 = TT_class.GenRankOneTT(n_p,d_p,1,scaling_m0);

%%%%% Create the initial R measurement scalar %%%%%
scaling_R =  1e-8;  
R = scaling_R;      
                    
%%%%% Create the initial Q TTM noise - set scaling to zero if unused  %%%%%
scaling_Q = 0; 
Q0 = TT_class.GenRankOneTT(n_p,2*d_p,2,scaling_Q);

%%%%% Create the initial P TTM covariance %%%%%
scaling_P0 = 1; %sigma^2
P0 = TT_class.GenRankOneTT(n_p,2*d_p,2,scaling_P0);

%%%%% Create the TTV of 1s vector %%%%%
TTVSumVector = TT_class.GenRankOneTT(n_p,d_p,1,1);

%%%%% Rank truncation for system TT's %%%%%
%Training
Trunc_Par.DefaultMaxR           = inf; 
Trunc_Par.DefaultMaxEps         = 0; 
Trunc_Par.RankTrunc_m           = inf; 
Trunc_Par.Eps_m                 = 0.001;
Trunc_Par.RankTrunc_P           = inf;
Trunc_Par.Eps_P                 = 0.001;
Trunc_Par.RankTrunc_C           = 30;
Trunc_Par.Eps_C                 = 0;
Trunc_Par.RankTrunc_S_k         = inf;
Trunc_Par.Eps_S_k               = 0.001;
Trunc_Par.RankTrunc_K_k         = 6;
Trunc_Par.Eps_K_k               = 0;

%Validation
Trunc_Par.RankTrunc_KernelRow   = Trunc_Par.RankTrunc_C ; 
Trunc_Par.Eps_KernelRow         = Trunc_Par.Eps_C   ;
Trunc_Par.MAXrank_cov_Y         = inf; 
Trunc_Par.MAXEps_cov_Y          = 0.001; 

%% Call the TTKF method to iterate over rows of the data matrix.
%%%% Explanation of function inputs:

tic
[TTKF_output, StabilityVecs] = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R,Q0,Trunc_Par,n_p,d_p,lambda,ConvCond,TTVSumVector);
toc

%% Initialize regression with TTKF 

alpha_TT = TTKF_output(1);
covariance_TT = TTKF_output(2);  


%% Validation Performance

[classifier_function,y_upperbound,y_lowerbound] = TTClassification(alpha_TT,covariance_TT,Trunc_Par,R,b,sig2,X_t,X_p,n_t,d_t,n_p,d_p,LSSVM);

%%
labels_validation_mean = sign(classifier_function);
labels_validation_upperbound = sign(y_upperbound);
labels_validation_lowerbound = sign(y_lowerbound);

figure
plot(labels_validation_mean)
hold on
plot(labels_validation_upperbound,'g-')
plot(labels_validation_lowerbound,'g-')

num_bounds_missclassification = sum((labels_validation_lowerbound~=labels_validation_mean)+(labels_validation_upperbound~=labels_validation_mean));
confidence_solution = 1-num_bounds_missclassification/(n_t^(d_t))


num_correct      = sum(labels_validation_mean == Labels_t);
num_incorrect    = length(labels_validation_mean)-num_correct;
percentage_wrong = num_incorrect/length(labels_validation_mean)
percentage_right = num_correct/length(labels_validation_mean)



