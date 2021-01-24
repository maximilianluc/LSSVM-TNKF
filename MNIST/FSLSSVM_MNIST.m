%% FSLSSVM for the MNIST dataset
clear all
close all
clc

%% Subset selections  
% p: practice / training
% t: test / validation

train_data = readtable('mnist_train.csv');
test_data  = readtable('mnist_test.csv');

%%Option 1: The input images to grayscale [0-1] by dividing by 255.
%X_p_all = train_data{:,2:end}/255;
%X_t_all = test_data{:,2:end}/255;

%%Option 2: The input images (rows) normalized
X_p_all = normalize(train_data{:,2:end},2);
X_t_all = normalize(test_data{:,2:end},2);


%The labels [0-9] 
Labels_p_all = train_data{:,1};
Labels_t_all = test_data{:,1};

bigger_5_p  = Labels_p_all(:,1) >= 5;  %label -1
smallereq_4_p = Labels_p_all(:,1) <= 4;  %label 1
Labels_p_all = bigger_5_p - smallereq_4_p;

bigger_5_t  = Labels_t_all(:,1) >= 5;  %label -1
smallereq_4_t = Labels_t_all(:,1) <= 4;  %label 1
Labels_t_all = bigger_5_t-smallereq_4_t;

n_p = 3;
d_p = 9;

n_t = 3;
d_t = 7;

X_p = X_p_all(1:n_p^d_p,:);
Labels_p = Labels_p_all(1:n_p^d_p,:);

%%% Can sort training data (note sorting has no influence)
% [Labels_p,I] = sort(Labels_p,'descend');
% X_p = X_p(I,:);

X_t = X_t_all(1:n_t^d_t,:); 
Labels_t = Labels_t_all(1:n_t^d_t,:);

%%% Can sort test data 
% [Labels_t,I_t] = sort(Labels_t,'descend');
% X_t = X_t(I_t,:);

%% initial values
%%% Here the initial values are

gam  = 0.05; %   best gamma around 0.00005 - 0.0005                                             
sig2 = 5;  %best sigma around 0.5
Nc= 800;

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









