%%
clear all
close all
clc

%% Identification Data

run('InitializeAdultData.m')
load('randperm_nonchanging.mat')

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
[Labels_p,I] = sort(Labels_p,'descend');
X_p = X_p(I,:);


X_t = X_unsorted((n_p^d_p)+1:(n_p^d_p)+(n_t^d_t)+1,:); 
Labels_t = Labels_unsorted((n_p^d_p)+1:(n_p^d_p)+(n_t^d_t)+1);


%% initial values
%%% Here the initial values are

gam  = 0.0015; %   best gamma around 0.00005 - 0.0005                                             
sig2 = 0.5;  %best sigma around 0.5
Nc= 1500;

%Z
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
b_p = 0; 
features_training = AFEm(X_p(sv,:),'RBF_kernel',sig2, X_p);
[W,b] = ridgeregress(features_training, Labels_p, gam); 
Y_training_pred = sign(features_training*W+b_p);
features_val = AFEm(X_p(sv,:),'RBF_kernel',sig2, X_t);
labels_validation = sign(features_val*W+b_p);



num_correct      = sum(labels_validation == Labels_t);
num_incorrect    = length(labels_validation)-num_correct;
percentage_wrong = num_incorrect/length(labels_validation)
percentage_right = num_correct/length(labels_validation)









