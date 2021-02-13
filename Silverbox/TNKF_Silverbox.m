%% The Run file for the Silverbox data set
clear all
close all
clc
echo off
profile off
profile on


%% Subset selections  
% p: practice / training
% t: test / validation

n_p= 2;
d_p= 15;
stepsize_p = 1;

n_t = 2;
d_t = 15;
stepsize_t = 1;


% y_id_window - the lag in y (outputs)
% u_id_window - the lag in u (inputs)
% delay_u     - the delay in the inputs
y_id_window = 5;  %paper   
u_id_window = 5;  %paper 
delay_u = 0;           

%load data set
load('SNLS80mV.mat')


%V1 - input - designed to be zero mean
%V2 - output - assumed to be zero mean
V1=V1-mean(V1); % Remove offset errors on the input measurements (these are visible in the zero sections of the input)
                % The input is designed to have zero mean
V2=V2-mean(V2); % Approximately remove the offset errors on the output measurements. 
                % This is an approximation because the silverbox can create itself also a small DC level               
                
                
begin_point_p = 40000;  % end at: begin_point_p+n_p^d_p-1
begin_point_t = 85000;  % lag=5  % end at: begin_point_t+n_t^d_t-1

testsize_p = n_p^d_p;
testsize_t = n_t^d_t;


%Create the lagged input and output data
i=1;
for lag = delay_u:u_id_window+delay_u-1  
    U_id_p(i,:) = (V1(1, begin_point_p-lag :    stepsize_p:   begin_point_p+(testsize_p)*stepsize_p -1-lag ));
    U_id_t(i,:) = (V1(1, begin_point_t-lag :    stepsize_t:   begin_point_t+(testsize_t)*stepsize_t -1-lag));
    i=i+1;
end

i=1;
for lag = 1:y_id_window
    Y_id_p(i,:) = (V2(1,  begin_point_p-lag:  stepsize_p  :begin_point_p+(testsize_p)*stepsize_p -1-lag));
    Y_id_t(i,:) = (V2(1,  begin_point_t-lag:    stepsize_t  :begin_point_t+(testsize_t)*stepsize_t -1-lag));
    i=i+1;
end

% Create the lagged training and test data
if exist('Y_id_p','var') == 1 &&  exist('Y_id_t','var') == 1
    X_p = [U_id_p' Y_id_p'];
    X_t = [U_id_t' Y_id_t'];
elseif exist('Y_id_p','var') == 0 &&  exist('Y_id_t','var') == 0
    X_p = U_id_p';
    X_t = U_id_t';
end


Y_p = V2(1,  begin_point_p:stepsize_p:begin_point_p+(testsize_p)*stepsize_p -1)';
Y_t = V2(1,  begin_point_t:stepsize_t:begin_point_t+(testsize_t)*stepsize_t -1)';

% Sort the data according to norm
%[~,I_p] = sort(Y_p,'descend');
[~,I_p] = sort(vecnorm(Y_p,1,2),'descend');
X_p = X_p(I_p,:);
Y_p = Y_p(I_p,:);

%[~,I_t] = sort(Y_t,'descend');
[~,I_t] = sort(vecnorm(Y_t,1,2),'descend');
X_t = X_t(I_t,:);
Y_t = Y_t(I_t,:);



b_p = mean(Y_p);
Y_p = Y_p-b_p;

%Standardize?


%% Kernel, Hyperparameters, NARX structure selections

%%% Specify the input and output data for validation and test sets

%type = 'function estimation';


%%% Assign kernel parameters
KernelFunc = 'RBF';      % 'Linear' or 'RBF' 
gam  = 0.085;   %                            
sig2 = 0.009  ; %5.19*lag^(-.5)

%% Early stopping

%%%%% Convergence conditions %%%%%
ConvCond.EarlyStopping                      = 'no';
ConvCond.FactorRemainingUncertainty         = 0;                          % relative to P0 = percentage - of squared sv's
ConvCond.FactorMinUncertaintChangeIter      = 0;                          % factor/percentage uncertainty change per iteration of the squared sv's
ConvCond.FactorMinUncertaintChangeNumIter   = 0;                          % after passing through 1% of rows, if the trace change stays below...
ConvCond.SumAlphaBound                      = 0;                          % total error of normalized ('preprocessed system')


%% Data initialization

%%% Create output data vector
y = Y_p;
Data_output_vec = [0;y];

%%% Create a LSSVM data structure
LSSVM.type = 'regression';       % 'classification' or 'regression'
LSSVM.KernelFunc = KernelFunc; 
LSSVM.Xp = X_p;
LSSVM.OutputVec = Data_output_vec;
LSSVM.gamma = gam;
LSSVM.sig2 = sig2; 


%% Intialization of Kalman system ("KF" structure)
% - Create and set parameters -- might have to change the approach. Now
% I first create matrices -> in the future create TT's directly.

%%%%% Forgetting factor - covariance TT iterations %%%%%
lambda = 1;

%%%%% Create the TTV of weight vector %%%%%
scaling_m0 = 0; % only the alpha weights
m0 = TT_class.GenRankOneTT(n_p,d_p,1,scaling_m0);

%%%%% Create the initial R measurement scalar %%%%%
scaling_R =  5*10^-7;   % What I have measured from the first 1800 samples
R = scaling_R;      % Variance value of R -> sigma^2 -- variance on Yp

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
Trunc_Par.Eps_m                 = 0.03;
Trunc_Par.RankTrunc_P           = 1;
Trunc_Par.Eps_P                 = 0;
Trunc_Par.RankTrunc_C           = inf;
Trunc_Par.Eps_C                 = 0.03;
Trunc_Par.RankTrunc_S_k         = 1;
Trunc_Par.Eps_S_k               = 0;
Trunc_Par.RankTrunc_K_k         = 1;
Trunc_Par.Eps_K_k               = 0;


%Validation
Trunc_Par.RankTrunc_KernelRow   = Trunc_Par.RankTrunc_C; 
Trunc_Par.Eps_KernelRow         = Trunc_Par.Eps_C ;
Trunc_Par.MAXrank_cov_Y         = inf; 
Trunc_Par.MAXEps_cov_Y          = 0.03; 

%% Call the TTKF method to iterate over rows of the data matrix.
%%%% Explanation of function inputs:

tic
[TTKF_output, StabilityVecs] = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R,Q0,Trunc_Par,n_p,d_p,lambda,ConvCond,TTVSumVector);
toc

%% Initialize regression with TTKF 

alpha_TT = TTKF_output(1);
covariance_TT = TTKF_output(2);  

%% Test Performance


[y_test,y_test_bounds] = TTRegression(alpha_TT,covariance_TT,Trunc_Par,R,b_p,sig2,X_t,X_p,n_t,d_t,n_p,d_p,LSSVM);
y_upperbound = y_test+3.*sqrt(y_test_bounds);
y_lowerbound = y_test-3.*sqrt(y_test_bounds);

%y_test = y_test - mean(y_test);


%% Unsort
X_p = X_p(I_p,:);
Y_p = Y_p(I_p,:);
y_training = y_training(I_p,:)
RMSE_training = sqrt((1/length(Y_p))*(sum((y_training-Y_p).^2)))


%%
X_t = X_t(I_t,:);
Y_t = Y_t(I_t,:);
y_test = y_test(I_t,:)
RMSE_test = sqrt((1/length(Y_t))*(sum((y_test-Y_t).^2)))
RMSE_conf_upper = sqrt((1/length(Y_t))*(sum((y_upperbound-Y_t).^2)))
RMSE_conf_lower = sqrt((1/length(Y_t))*(sum((y_lowerbound-Y_t).^2)))


%% figure(2)
h = figure
plot(Y_t,'b-')
hold on
plot(y_test,'r-')
hold on


%RMSE_TT_training
RMSE_validation 
data_t = iddata(Y_t,X_t,1);
data_v = iddata(y_test,X_t,1);
[~,fit,~] = compare(data_v,data_t)

