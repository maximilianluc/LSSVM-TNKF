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
nb = 250;


%%
tic
[V, D] = eign(X_p, 'RBF_kernel', sig2, nb);
toc
diagD = diag(D);
alpha = gam*(Y_p_nomean - (V*inv((1/gam)*eye(length(D))+diagD*(V'*V)))*diagD*V'*Y_p_nomean);
 
[Ypred_training, Zp] = simlssvm({X_p,Y_p_nomean,'function estimation',gam,sig2,'RBF_kernel','original'}, {alpha,b_p}, X_p);
[Ypred_validation, Zp] = simlssvm({X_p,Y_p_nomean,'function estimation',gam,sig2,'RBF_kernel','original'}, {alpha,b_p}, X_t);

RMSE_training_Nystrom = sqrt((1/length(Y_p))*(sum((Ypred_training-Y_p).^2)))
RMSE_val_Nystrom = sqrt((1/length(Y_t))*(sum((Ypred_validation-Y_t).^2)))

plot(X_t,Y_t,'b.')
hold on
plot(X_t,Ypred_validation,'r*')
grid on

data_t = iddata(Y_t,X_t);
data_v = iddata(Ypred_validation,X_t);
RMSE_training_Nystrom
RMSE_val_Nystrom
[~,fit,~] = compare(data_v,data_t)
