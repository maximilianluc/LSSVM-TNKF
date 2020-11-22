%%
clear all
close all
clc


%% Identification Data
n=4,d=6;
run('IdentificationDataInit.m')
model_FSLSSVM = initlssvm(X_p,Y_p,'function estimation',[],[],'RBF_kernel','preprocess');
[Xp, Yp] = prelssvm(model_FSLSSVM, X, Y);


%% Grid search for sig2 and number of support vectors


%%% Grid search to determine "good" FS-LSSVM hyperparameters (number of support
%%% vectors and sig2). Manually adjust them in initial values section

%             numSV_guess_vec = 25:25:500             ;
%             sig2s_guess_vec = 0.0001:0.0002:0.002    ;
%             trials_per_hyperparamters = 5           ;
%             [Num_SV, sig2] = GridSearch_sig2_numSV(Xp,Yp,sig2s_guess_vec, numSV_guess_vec, trials_per_hyperparamters)



%% initial values
%%% Here the initial values are


sig2=0.005;%3e-4;
Nc=500;

%
% load data X and Y, ’capacity’ and the kernel parameter ’sig2’
sv = 1:Nc;
max_c = -10000; %was -inf, not sure what a good value is...
tic 
for i=1:size(Xp,1)
    i
    replace = ceil(rand.*Nc);
    subset = [sv([1:replace-1 replace+1:end]) i];
    crit = kentropy(Xp(subset,:),'RBF_kernel',sig2);
    if max_c <= crit, max_c = crit; sv = subset; end
end
toc
%%

features = AFEm(Xp(sv,:),'RBF_kernel',sig2, Xp);
[Cl3, gam_optimal] = bay_rr(features,Yp,1,3);
[W,b] = ridgeregress(features, Yp, gam_optimal); 

Y_fslssvm = features*W+b;

RMSE_FSLSSVM = sqrt((1/length(Yt))*(sum((Y_fslssvm-Yt).^2)))
