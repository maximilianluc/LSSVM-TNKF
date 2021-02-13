function [classifier_function,y_upperbound,y_lowerbound] = TTClassification(alpha_TT,covariance_TT,Trunc_Par,meas_noise,bias,sig2,X_test,X_prac,n_test,d_test,n_prac,d_prac,LSSVM)

% validation data must be row-ordered

length_k = size(X_test,1);
y_validation = zeros(n_test^d_test,1);
y_upperbound = zeros(n_test^d_test,1);
y_lowerbound = zeros(n_test^d_test,1);
y_variance   = zeros(n_test^d_test,1); 
classifier_function = zeros(n_test^d_test,1); 

 %%% Instead of programming hadamard, i'll just do .* with the vectors
    % - computationally this does not require that much extra time, but it isnt optimal.
    alpha_vec = reshape(ContractTTtoTensor(alpha_TT),[n_prac^d_prac 1]);
    alpha_times_labels =  alpha_vec.*LSSVM.labels;
    alpha_k_y_k = TT_class(alpha_times_labels,n_prac,d_prac,0.001,20,1);
    covariance_TT = TTRounding(covariance_TT,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);

for k=1:length_k
    k_ = k
    diff = zeros(n_prac^d_prac,size(X_prac,2));
    diff(1:n_prac^d_prac,:) = X_test(k,:)-X_prac;
    
    if strcmp(LSSVM.KernelFunc,'Linear')
        Linear_row =  sum(diff.*diff,2)' ;
        KernelRow = TT_class(Linear_row,n_prac,d_prac,Trunc_Par.Eps_KernelRow,Trunc_Par.RankTrunc_KernelRow,1);
    elseif strcmp(LSSVM.KernelFunc,'RBF')
        RBF_row = (exp( -((sum(diff.*diff,2))'./(2*sig2))) );
        KernelRow = TT_class(RBF_row,n_prac,d_prac,Trunc_Par.Eps_KernelRow,Trunc_Par.RankTrunc_KernelRow,1); 
    end
    
    
    cov_y_RC                = ContractTwoTT(covariance_TT,KernelRow,3,2);
    cov_y_RC                = TTRounding(cov_y_RC,Trunc_Par.MAXEps_cov_Y,Trunc_Par.MAXrank_cov_Y);
    cov_y_LC                = ContractTwoTT(KernelRow,cov_y_RC,2,2);
    S_k_scalar              = ContractTTtoTensor(cov_y_LC);
    y_variance(k)           = S_k_scalar + meas_noise;
    
    
    classifier_function(k) = ContractTTtoTensor(ContractTwoTT(alpha_k_y_k,KernelRow,2,2))+bias; 
    y_upperbound(k) = classifier_function(k)+3*y_variance(k);
    y_lowerbound(k) = classifier_function(k)-3*y_variance(k);
    
    
    
end
end

