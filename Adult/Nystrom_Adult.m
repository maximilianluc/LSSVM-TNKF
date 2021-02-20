%%
clear all
close all
clc


%% Identification Data

run('InitializeAdultData.m')
load('randperm_nonchanging.mat')

% RANDPERM = randperm(size(adultdata,1));
% randperm_nonchanging=RANDPERM;
X_unsorted = X(randperm_nonchanging,:);              % as to mix up the data first
Labels_unsorted = Labels(randperm_nonchanging,:);    % as to mix up the data first

% RANDPERM = randperm(size(adultdata,1));
% X = X(RANDPERM,:);
% Labels = Labels(RANDPERM,:);

n_p = 3;
d_p = 9;

n_t = 3;
d_t = 7;


X_p = X_unsorted(1:n_p^d_p,:);
Labels_p = Labels_unsorted(1:n_p^d_p,:);
%%% Sort training data
%[Labels_p,I] = sort(Labels_p,'descend');
%X_p = X_p(I,:);


X_t = X_unsorted((n_p^d_p)+1:(n_p^d_p)+(n_t^d_t)+1,:); 
Labels_t = Labels_unsorted((n_p^d_p)+1:(n_p^d_p)+(n_t^d_t)+1);

%% initial values
%%% Here the initial values are

gam  = 0.0015; %   best gamma around 0.00005 - 0.0005                                             
sig2 = 0.5;  %best sigma around 0.5
nb = 50;

S = 3^6;
RandPermutation = randperm(length(X_p))
Subset = X_p(RandPermutation(1:S),:)
Subset_labels = Labels_p(RandPermutation(1:S),:)

tic
[V, D] = eign(Subset, 'RBF_kernel', sig2, nb);
diagD = diag(D);
alpha = gam*(Subset_labels - (V*inv((1/gam)*eye(length(D))+diagD*(V'*V)))*diagD*V'*Subset_labels);
toc 

b_p=0;
[Ylabels_training, Zp] = simlssvm({Subset,Subset_labels,'c',gam,sig2,'RBF_kernel','o'}, {alpha,b_p}, Subset);
[Ylabels_validation, Zp] = simlssvm({Subset,Subset_labels,'c',gam,sig2,'RBF_kernel','o'}, {alpha,b_p}, X_t);


num_correct      = sum(Ylabels_validation == Labels_t);
num_incorrect    = length(Ylabels_validation)-num_correct;
percentage_wrong = num_incorrect/length(Ylabels_validation)
percentage_right = num_correct/length(Ylabels_validation)
