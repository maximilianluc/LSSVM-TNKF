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


%% initial values

gam  = 0.085  ;                              
sig2 = 0.009  ; 
Nc= 500;

%
% load data X and Y, ’capacity’ and the kernel parameter ’sig2’
sv = 1:Nc;
max_c = -inf; %was -inf, not sure what a good value is...
tic 
for i=1:size(X_p,1)
    i
    replace = ceil(rand.*Nc);
    subset = [sv([1:replace-1 replace+1:end]) i];
    crit = kentropy(X_p(subset,:),'RBF_kernel',sig2);
    if max_c <= crit, max_c = crit; sv = subset; end
end
toc
%%

features_training = AFEm(X_p(sv,:),'RBF_kernel',sig2, X_p);
[W,b] = ridgeregress(features_training, Y_p, gam); 
Y_training_pred = features_training*W+b_p;
features_test = AFEm(X_p(sv,:),'RBF_kernel',sig2, X_t);
Y_test_pred = features_test*W+b_p;



%% Unsort
X_p = X_p(I_p,:);
Y_p = Y_p(I_p,:);
Y_training_pred = Y_training_pred(I_p,:);
RMSE_training_FSLSSVM = sqrt((1/length(Y_p))*(sum((Y_training_pred-Y_p).^2)))

X_t = X_t(I_t,:);
Y_t = Y_t(I_t,:);
Y_test_pred = Y_test_pred(I_t,:);
RMSE_test_FSLSSVM = sqrt((1/length(Y_t))*(sum((Y_test_pred-Y_t).^2)))


%%

plot(Y_t,'b-')
hold on
plot(Y_test_pred,'r-')
grid on


data_t = iddata(Y_t,X_t);
data_v = iddata(Y_test_pred,X_t);
[~,fit,~] = compare(data_v,data_t)



