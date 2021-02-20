%%
clear all
close all
clc


%% Identification Data

n=2;
d=16;

%%% Practice data
label1 = ones(0.5*(n^d),1);
label2 = -ones(0.5*(n^d),1);
labels = [label1;label2];
Labels_p = labels;
[Spiral1_Xp,Spiral1_Yp] = SpiralFunction(6,180,n,d,0);
[Spiral2_Xp,Spiral2_Yp] = SpiralFunction(6,180,n,d,pi);

X_p = ([Spiral1_Xp,Spiral1_Yp;Spiral2_Xp,Spiral2_Yp]);

%%% Test data                             %180
[Spiral1_Xt,Spiral1_Yt] = SpiralFunction(6,180,n,d,0);
[Spiral2_Xt,Spiral2_Yt] = SpiralFunction(6,180,n,d,pi);

X_t = ([Spiral1_Xt,Spiral1_Yt;Spiral2_Xt,Spiral2_Yt]);

labels_correct_validation = [ones(0.5*(n^(d)),1);-ones(0.5*(n^(d)),1)]; %already sorted!
Labels_t = labels_correct_validation; 
figure(1)
plot(X_p(:,1),X_p(:,2),'g*')
hold on
plot(X_t(:,1),X_t(:,2),'k.')



%% initial values
%%% Here the initial values are

gam  = 0.05;                                                    
sig2 = 5e-5; 
nb = 500;

S = n^15;
RandPermutation = randperm(length(X_p))
Subset = X_p(RandPermutation(1:S),:)
Subset_labels = Labels_p(RandPermutation(1:S),:)

tic
[V, D] = eign(Subset, 'RBF_kernel', sig2, nb);
diagD = diag(D);
alpha = gam*(Subset_labels - (V*inv((1/gam)*eye(length(D))+diagD*(V'*V)))*diagD*V'*Subset_labels);
toc 



%%
b_p=0;
[Ylabels_training, Zp] = simlssvm({Subset,Subset_labels,'c',gam,sig2,'RBF_kernel'}, {alpha,b_p}, Subset);
[Ylabels_validation, Zp] = simlssvm({Subset,Subset_labels,'c',gam,sig2,'RBF_kernel'}, {alpha,b_p}, X_t);


%%
b_p=0;
[Ylabels_training, Zp] = simlssvm({X_p,Labels_p,'c',gam,sig2,'RBF_kernel'}, {alpha,b_p}, X_p);
[Ylabels_validation, Zp] = simlssvm({X_p,Labels_p,'c',gam,sig2,'RBF_kernel'}, {alpha,b_p}, X_t);


num_correct      = sum(Ylabels_validation == Labels_t);
num_incorrect    = length(Ylabels_validation)-num_correct;
percentage_wrong = num_incorrect/length(Ylabels_validation)
percentage_right = num_correct/length(Ylabels_validation)

figure(3) %validation performance
idx_label1 = Ylabels_validation==1;
idx_label2 = Ylabels_validation==-1;
plot(X_t(idx_label1,1),X_t(idx_label1,2),'b.',X_t(idx_label2,1),X_t(idx_label2,2),'r.')
axis([-0.75 0.75 -0.75 0.75])



