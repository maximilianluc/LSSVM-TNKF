%% The Run file for the noisy sinc
clear all
close all
clc

%% Identification Data ----------------------------------------------
n_p = 2;
d_p = 14;
n_t = 2;
d_t = 13;
noise_sigma = 1e-1; 

X_p = linspace(-2.2,2.2,n_p^d_p)';
Y_p = sinc(X_p) +normrnd(0,noise_sigma,size(X_p,1),1);

X_t = linspace(-2.2,2.2,n_t^d_t)';
Y_t = sinc(X_t) +normrnd(0,noise_sigma,size(X_t,1),1);

%% Training ----------------------------------------------

%%% Specify the input and output data for validation and test sets

%%% Assign kernel parameters
KernelFunc = 'RBF';
gam        = 0.005;                                                   
sig2       = 0.005;   

%%% Convergence conditions (Early stopping) 
ConvCond.EarlyStopping = 'no';
ConvCond.FactorRemainingUncertainty         = 0.5;%                        % relative to P0 = percentage - of squared sv's
ConvCond.FactorMinUncertaintChangeIter      = 1/(n_p^d_p);                 % factor/percentage uncertainty change per iteration of the squared sv's
ConvCond.FactorMinUncertaintChangeNumIter   = 5;                           % after passing through 1% of rows, if the trace change stays below...
ConvCond.SumAlphaBound                      = 0.00000001;  %0.0001;        % total error of normalized ('preprocessed system')



%% Data initialization ----------------------------------------------

%%% Subtract mean from output data
b_p = mean(Y_p,1);
Y_p_nomean = Y_p-b_p;
b_t = mean(Y_t,1);
Y_t_nomean = Y_t-b_t;

%%% Create output data vector
y = Y_p_nomean;
Data_output_vec = [0;y];

%%% Create a LSSVM data structure
LSSVM.type = 'regression';                                                 % 'classification' or 'regression'
LSSVM.KernelFunc = KernelFunc; 
LSSVM.Xp = X_p;
LSSVM.OutputVec = Data_output_vec;
LSSVM.gamma = gam;
LSSVM.sig2 = sig2; 

%% Intialization of Kalman filter system ("KF" structure) ----------------------------------------------

%%%%% Parameters for the design %%%%%
lambda = 1;
Trunc_Par.DefaultMaxR = inf;

%%%%% Create the TTV of weight vector %%%%%
scaling_m0 = 0; % only the alpha weights
m0 = TT_class.GenRankOneTT(n_p,d_p,1,scaling_m0);

%%%%% Create the initial R measurement scalar %%%%%
scaling_R =  noise_sigma^2;     
R = scaling_R;                 

%%%%% Create the initial Q TTM noise %%%%%
scaling_Q = 0; 
Q0 = TT_class.GenRankOneTT(n_p,2*d_p,2,scaling_Q);

%%%%% Create the initial P TTM covariance %%%%%
scaling_P0 = 1;                                                              %sigma^2
P0 = TT_class.GenRankOneTT(n_p,2*d_p,2,scaling_P0);

%%%%% Create the TTV of 1s vector %%%%%
TTVSumVector = TT_class.GenRankOneTT(n_p,d_p,1,1);

%%%%% Rank truncation for system TT's %%%%%
Trunc_Par.DefaultMaxR           = inf; 
Trunc_Par.DefaultMaxEps         = 0; 
Trunc_Par.RankTrunc_m           = inf; 
Trunc_Par.Eps_m                 = 0;
Trunc_Par.RankTrunc_P           = inf;
Trunc_Par.Eps_P                 = 0.015;
Trunc_Par.RankTrunc_C           = inf;
Trunc_Par.Eps_C                 = 0.003;
Trunc_Par.RankTrunc_S_k         = inf;
Trunc_Par.Eps_S_k               = 0.1;
Trunc_Par.RankTrunc_K_k         = inf;
Trunc_Par.Eps_K_k               = 0.0075;

%Validation
Trunc_Par.RankTrunc_KernelRow   = Trunc_Par.RankTrunc_C ; 
Trunc_Par.Eps_KernelRow         = Trunc_Par.Eps_C;
Trunc_Par.MAXrank_cov_Y         = inf; 
Trunc_Par.MAXEps_cov_Y          = 0.0015; 




%% Call the TNKF method for training ----------------------------------------------

%%% Call the TTKF method to iterate over rows of the data matrix.
tic
[TTKF_output, StabilityVecs] = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R,Q0,Trunc_Par,n_p,d_p,lambda,ConvCond,TTVSumVector);
toc

%%% Assign function outputs
alpha_TT = TTKF_output(1);
covariance_TT = TTKF_output(2);  



%% Validation Performance - what its all about  ----------------------------------------------

[y_validation,y_val_variance] = TTRegression(alpha_TT,covariance_TT,Trunc_Par,R,b_p,sig2,X_t,X_p,n_t,d_t,n_p,d_p,LSSVM);

RMSE_validation = sqrt((1/length(Y_t))*(sum((y_validation-Y_t).^2)));
sum_alphas = StabilityVecs(end,2);




%% Plotting of the TNKF regression/classification
y_upperbound = y_validation+3.*sqrt(y_val_variance);
y_lowerbound = y_validation-3.*sqrt(y_val_variance);




figure(15)
plot(X_t,Y_t,'b.')
hold on
title('model TT')
plot(X_t,y_validation,'r*')

%%
clear figure(16)
figure(16)
plot_step_size = 4*64;
plot(X_t,Y_t,'b.')
hold on
title('model TT')
plot(X_t,y_validation,'r.','MarkerSize',20,'LineWidth',1)
plot(X_t(1:plot_step_size:end),real(y_upperbound(1:plot_step_size:end)),'-.go','MarkerSize',9,'LineWidth',2.5)
plot(X_t(1:plot_step_size:end),real(y_lowerbound(1:plot_step_size:end)),'-.go','MarkerSize',9,'LineWidth',2.5)
grid on
title('Test performance TNKF - noisy sinc')
xlabel('x')
ylabel('y')
legend('data','mean','confidence bounds')

data_t = iddata(Y_t,X_t);
data_v = iddata(y_validation,X_t);

RMSE_validation;
[~,fit,~] = compare(data_v,data_t);



