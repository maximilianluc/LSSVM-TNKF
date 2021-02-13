%%
clear all
close all
clc

%% Subset selections  
% p: practice / training
% t: test / validation

n=2;d=16;
step = 4; %(validation is every fourth point of the practice spiral)

%%% Practice data
label1 = ones(0.5*(n^d),1);
label2 = -ones(0.5*(n^d),1);
Labels_p = [label1;label2];
[Spiral1_Xp,Spiral1_Yp] = SpiralFunction(6,180,n,d,0);
[Spiral2_Xp,Spiral2_Yp] = SpiralFunction(6,180,n,d,pi);

X_p = ([Spiral1_Xp,Spiral1_Yp;Spiral2_Xp,Spiral2_Yp]);
%%% Test data                             %180
%[Spiral1_Xt,Spiral1_Yt] = SpiralFunction(6,180,n,d,0);
%[Spiral2_Xt,Spiral2_Yt] = SpiralFunction(6,180,n,d,pi);

%X_t = ([Spiral1_Xt,Spiral1_Yt;Spiral2_Xt,Spiral2_Yt]);
X_t = ([Spiral1_Xp(1:step:end),Spiral1_Yp(1:step:end);Spiral2_Xp(1:step:end),Spiral2_Yp(1:step:end)]);

Labels_t = [ones(0.5*(length(X_t)),1);-ones(0.5*(length(X_t)),1)]; %already sorted!

figure(1)
plot(X_p(:,1),X_p(:,2),'g*')
hold on
plot(X_t(:,1),X_t(:,2),'k.')

% plot(Spiral1_X,Spiral1_Y,'b*'),hold on,
% plot(Spiral2_X,Spiral2_Y,'r*')



%% initial values
%%% Here the initial values are

gam =0.0045;
sig2=5e-5;
Nc=800;
%% 
%
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
labels_training = sign(features_training*W+b_p);
features_val = AFEm(X_p(sv,:),'RBF_kernel',sig2, X_t);
labels_validation = sign(features_val*W+b_p);


%% Training performance 
num_correct_p      = sum(labels_training == Labels_p);
num_incorrect_p    = length(labels_training)-num_correct_p;
percentage_wrong_p = num_incorrect_p/length(labels_training)
percentage_right_p = num_correct_p/length(labels_training)


figure(3) %validation performance
idx_label1_p= labels_training==1;
idx_label2_p= labels_training==-1;
idx_label3_p= labels_training==0;

plot(X_p(idx_label1_p,1),X_p(idx_label1_p,2),'b.',X_p(idx_label2_p,1),X_p(idx_label2_p,2),'r.')
hold on
plot(X_p(idx_label3_p,1),X_p(idx_label3_p,2),'m.')


