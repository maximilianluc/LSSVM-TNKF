%%
clear all
close all
clc


%% Identification Data
n_p = 2;
d_p = 14;
n_t = 2;
d_t = 13;

noise_sigma = 0.1; 

X_p = linspace(-2.2,2.2,n_p^d_p)';
Y_p = sinc(X_p) +normrnd(0,noise_sigma,size(X_p,1),1);

X_t = linspace(-2.2,2.2,n_t^d_t)';
Y_t = sinc(X_t) +normrnd(0,noise_sigma,size(X_t,1),1);


b_p = mean(Y_p,1);
Y_p_nomean = Y_p-b_p;

b_t = mean(Y_t,1);
Y_t_nomean = Y_t-b_t;

[~,I_p] = sort(X_p,'ascend');
[~,I_t] = sort(X_t,'ascend');

X_p_perm = X_p(I_p);
Y_p_perm = Y_p_nomean(I_p);

X_t_perm = X_t(I_t);
Y_t_perm = Y_t_nomean(I_t);

X_p = X_p_perm;
Y_p_nomean = Y_p_perm;
X_t = X_t_perm;
Y_t_nomean = Y_t_perm;


%% initial values
%%% Here the initial values are

gam  = 0.005;                                                  
sig2 = 0.005;  
Nc= 500


% load data X and Y, ’capacity’ and the kernel parameter ’sig2’
sv = 1:Nc;
max_c = -inf; 
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
[W,b] = ridgeregress(features_training, Y_p_nomean, gam); 
Y_training_pred = features_training*W+b_p;
features_val = AFEm(X_p(sv,:),'RBF_kernel',sig2, X_t);
Y_val_pred = features_val*W+b_p;


RMSE_training_FSLSSVM = sqrt((1/length(Y_p))*(sum((Y_training_pred-Y_p).^2)))
RMSE_val_FSLSSVM = sqrt((1/length(Y_t))*(sum((Y_val_pred-Y_t).^2)))



plot(X_t,Y_t,'b.')
hold on
plot(X_t,Y_val_pred,'r*')
grid on


data_t = iddata(Y_t,X_t);
data_v = iddata(Y_val_pred,X_t);
RMSE_training_FSLSSVM
RMSE_val_FSLSSVM
[~,fit,~] = compare(data_v,data_t)



