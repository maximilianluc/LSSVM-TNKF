classdef TTKalmanFilter < TT_class
    %TTKALMANFILTER - Tensor train Kalman filter
    %   This is a class designed to contain the main methods//properties of
    %   the TT Kalman filter wrt the LSSVM problem.
    
    properties
        A = TT_class();       % the A system matrix -> in TT object form
        m_meas = TT_class();  % the A system matrix -> in TT object form (Changes per row/iteration in data matrix)
        P_meas = TT_class();  % the covariance matrix -> in TT object form (changes per iteration)
        Q = TT_class();
        
        
    end
    
    methods(Static)
        
        
        %%
        function [TTKF_output, StabilityVecs] = TTKF_method(LSSVM,m0_TT,P0_TTM,R0,Q0_TTM,Trunc_Par,n,d,lambda,ConvCond,TTVSumVector)
            
            m_meas  = m0_TT;
            P_meas  = P0_TTM;
            Q       = Q0_TTM;
            R       = R0;
            n       = n;
            d       = d;
            %Peig = zeros(n^d,n^d);
            sum_alphas = zeros(n^d+1,1);
            P_meas_squared_FrobNorm = zeros(n^d+1,1);
            
            %%%% Initialize
            P_meas_squared_FrobNorm(1) = (P_meas.SV_squared); %sqrt
            
            for k=1:n^d+1  %+1 because first row is "ones"
                
                %% Description of this for loop
                k 
                
                
                %% Analysis
                if strcmp(ConvCond.EarlyStopping, 'yes')
                    if strcmp(ConvCond.type,'regression')
                        sum_alphas_vec(k)         = ContractTTtoTensor(ContractTwoTT(m_meas,TTVSumVector,2,2));
                    elseif strcmp(ConvCond.type,'classification')
                        alpha_vec = reshape(ContractTTtoTensor(m_meas),[n^d 1]);
                        alpha_times_labels =  alpha_vec.*LSSVM.labels;
                        sum_alphas_vec(k) = sum(alpha_times_labels);
                    end
                end
                P_meas_squared_FrobNorm(k)    = (P_meas.SV_squared); %sqrt for frob norm-> squared forb norm=sum of SVs squared
                %
%                                 P = ContractTTtoTensor(P_meas);
%                                 P = reshape(P,[n^d n^d]);
%                 %
%                 %
%                                 Peig(:,k) = sort(eig(P));
%                                 
                
                %% Convergence Checks
                
                if k>1
                    if strcmp(ConvCond.EarlyStopping,'yes')
                        ConvergenceStatus = TTKalmanFilter.ConvergenceCheck(ConvCond,sum_alphas_vec(k),P_meas_squared_FrobNorm,k);
                    else
                        ConvergenceStatus =0;
                    end
                    
                    if ConvergenceStatus==1
                        
                        % not accirate
                        
%                         for index=0:1:(n^d)-1
%                             SelectionVec = [zeros(1,index) 1 zeros(1,(n^d)-1-index)];
%                             TT_SelVec    = TT_class(SelectionVec,n,d,0,inf,1);
%                             Var_right    = ContractTwoTT(P_meas,TT_SelVec,3,2);
%                             Var_left     = ContractTwoTT(TT_SelVec,Var_right,2,2);
%                             Variance(index+1,1)     = ContractTTtoTensor(Var_left);
%                         end
                        
                        
                        for i=1:25
                            disp('Possibly converged! Party party party!')
                            
                        end
                        pause(3)
                        TTKF_output = [m_meas, P_meas];
                        StabilityVecs = [P_meas_squared_FrobNorm sum_alphas];
                        %VarianceAlphas = Variance;  - not accurate
                        return
                    end
                    
                end
                
                
                
                %% Kalman Filter Prediction Step in TT Form
                
                %%%%% STEP 1
                % State update, prediction next time step is equal to previous state measured.
                % A_TT has ranks 1, so contraction have no affect, so not included;
                
                m_pred = m_meas;
                
                
                %   m_pred = TTRounding(m_pred,Trunc_Par.Eps_m,Trunc_Par.RankTrunc_m);
                
                
                %%%%% STEP 2
                % Covariance updated.
                P_pred  = P_meas;
                P_pred.Cores{P_pred.NumCores} = P_pred.Cores{P_pred.NumCores}*(lambda);
                P_pred = TTRounding(P_pred,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);
                
                if P_pred.MaxRank > Trunc_Par.DefaultMaxR
                    error('Large ranks P_pred')
                end
                
                
                %% Kalman Filter Update Step in TT Form
                %%%%% STEP 3
                % Transform the row of the LSSVM matrix to TT form
                
                if k==1 
                    if strcmp(LSSVM.type,'regression')
                        Trunc_Par.Eps_C0 = 0;
                        Trunc_Par.RankTrunc_C0 = 1;
                        C_k = TT_class(ones(1,n^d),n,d,Trunc_Par.Eps_C0,Trunc_Par.RankTrunc_C0,1);
                    end
                    if strcmp(LSSVM.type,'classification')
                        C_k = TT_class(LSSVM.labels,n,d,Trunc_Par.Eps_C,Trunc_Par.RankTrunc_C,1);
                    end
                elseif k>1
                    i = k-1; % because Xp does not have first row "ones" 
                    if strcmp(LSSVM.KernelFunc,'Linear')
                        Kernel_row = LinearKernelMax(LSSVM.Xp,LSSVM.gamma,n,d,i);
                    elseif strcmp(LSSVM.KernelFunc,'RBF')
                        Kernel_row = RBFKernelMax(LSSVM.Xp,LSSVM.gamma,LSSVM.sig2,n,d,i);
                    end
                    C_k = TT_class(Kernel_row,n,d,Trunc_Par.Eps_C,Trunc_Par.RankTrunc_C,1);
                end
                
                
                
                
                %                 if k>2
                %
                %                     CmatrixTTM = OuterProductTwoTTV(C_k,C_k);
                %                     TTMupdate = ContractTwoTT(CmatrixTTM,P_meas,3,2);
                %                     TTMupdate = TTRounding(TTMupdate,0,inf);
                %
                %                     Skupdate = ContractTwoTT(C_k,P_meas,2,2);
                %                     Skupdate = ContractTwoTT(Skupdate,C_k,2,2);
                %                     Skupdate = ContractTTtoTensor(Skupdate);
                %                     Skupdate = S_k_scalar + R;
                %
                %
                %                     TTMupdate.Cores{TTMupdate.NumCores} = TTMupdate.Cores{TTMupdate.NumCores}*(1/Skupdate);
                %                     UpdateTTM(:,:,k-1) = reshape(ContractTTtoTensor(TTMupdate),[n^d n^d]);
                %
                %                 end
                
                
                
                
                %%%%% STEP 4
                % Find the prediction error
                v_k = LSSVM.OutputVec(k) - ContractTTtoTensor(ContractTwoTT(C_k,m_pred,2,2));
                
                %%%%% STEP 5
                % Find the measurement covariance (S_k) - multiple steps
                S_k_RC              = ContractTwoTT(P_pred,C_k,3,2);
%                 if S_k_RC.MaxRank > Trunc_Par.DefaultMaxR
%                    S_k_RC              = TTRounding(S_k_RC,Trunc_Par.MAXEps,Trunc_Par.MAXrank);
%                 end
                S_k_RC              = TTRounding(S_k_RC,Trunc_Par.Eps_S_k,Trunc_Par.RankTrunc_S_k);
                S_k_LC              = ContractTwoTT(C_k,S_k_RC,2,2);
                S_k_scalar          = ContractTTtoTensor(S_k_LC);
                S_k                 = S_k_scalar + R;                   % scalar
                if S_k_RC.MaxRank > Trunc_Par.DefaultMaxR   
                    error('Large rank S_k ranks')
                end
                
                
                %%%%% STEP 6
                % Compute the Kalman gain (K_k)
                K_k_LC                                 = ContractTwoTT(P_pred,C_k,3,2);
                %   K_k_LC                                 = TTRounding(K_k_LC,Trunc_Par.Eps_K_k,Trunc_Par.RankTrunc_K_k);
                K_k_LC.Cores{K_k_LC.NumCores}          = K_k_LC.Cores{K_k_LC.NumCores}.*(1/S_k);
                K_k                                    = K_k_LC ;
                K_k =  TTRounding(K_k,Trunc_Par.Eps_K_k,Trunc_Par.RankTrunc_K_k);
                if K_k.MaxRank > Trunc_Par.DefaultMaxR
                    error('Large rank K_k ranks')
                end
                
                
                %%%%% STEP 7
                % compute the measured state (m_meas) by looking at the measured output
                KG = K_k;
                KG.Cores{KG.NumCores} = KG.Cores{KG.NumCores} * v_k;
                
                m_meas      = Add2TTs(m_pred,KG); %m_pred + K_k*v_k
                m_meas      = TTRounding(m_meas,Trunc_Par.Eps_m,Trunc_Par.RankTrunc_m);
                
                if m_meas.MaxRank > Trunc_Par.DefaultMaxR
                    error('Large rank m_meas ranks')
                end
                
                %%%%% STEP 8
                % compute the measured state covariance (P_meas) by looking at the
                % measured output
                
                K_OutProd  = OuterProductTwoTTV(K_k,K_k);
                K_OutProd  = TTRounding(K_OutProd,Trunc_Par.Eps_K_k,Trunc_Par.RankTrunc_K_k);
                K_OutProd.Cores{K_OutProd.NumCores} = K_OutProd.Cores{K_OutProd.NumCores}.*(-S_k); %last core has norm -> multiply with S
                
%                 if K_OutProd.MaxRank > Trunc_Par.DefaultMaxR
%                     k
%                     error('Large rank K_out ranks')
%                 end
                
                P_meas = Add2TTs(P_pred,K_OutProd);
                P_meas = TTRounding(P_meas,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);
                
                if P_meas.MaxRank > Trunc_Par.DefaultMaxR
                    error('Large rank P_meas ranks')
                end
                
                %%% Find the confidence bounds on alpha(k)
                
                if k==n^d+1
                        
                    % inaccurate to do for alphas. 
%                     for index=0:1:(n^d)-1
%                         SelectionVec = [zeros(1,index) 1 zeros(1,(n^d)-1-index)];
%                         TT_SelVec    = TT_class(SelectionVec,n,d,0,inf,1);
%                         Var_right    = ContractTwoTT(P_meas,TT_SelVec,3,2);
%                         Var_left     = ContractTwoTT(TT_SelVec,Var_right,2,2);
%                         Variance(index+1,1)     = ContractTTtoTensor(Var_left);
%                     end
                    disp('Finished iterations')
                end
                
                %                 if k==size(LSSVM.Matrix,1)
                %                     k=k+1;
                %                     sum_alphas(k)         = ContractTTtoTensor(ContractTwoTT(m_meas,SumVector,2,2));
                %                     P_meas_squared_FrobNorm(k)    = P_meas.SV_squared;
                %                     P = ContractTTtoTensor(P_meas);
                %                     P = reshape(P,[n^d n^d]);
                %                     Peig(:,k) = sort(eig(P));
                %                 end
                
                
                
                % Make P symmetric
                %                                     P_meas = ContractTTtoTensor(P_meas);
                %                                     P_meas = reshape(P_meas,[n^d n^d]);
                %                                     P_meas = 0.5*(P_meas+P_meas');
                %                                     P_meas = TT_class(P_meas,n,2*d,0,inf,2);
                %
                %
                %
                
                
                %if k == 1 || k == round(0.25*n^d) || k == round(0.5*n^d) || k == round(0.75*n^d) || k==n^d
                %                        figure(k+1)
                %                        image(P_matrix,'CDataMapping','scaled'), colorbar
                %                        pause(0.5)
                %end
                
                %                 if k==size(LSSVM.Matrix,1)
                %
                %                     CmatrixTTM = OuterProductTwoTTV(C_k,C_k);
                %                     TTMupdate = ContractTwoTT(CmatrixTTM,P_meas,3,2);
                %                     TTMupdate = TTRounding(TTMupdate,0,inf);
                %
                %                     Skupdate = ContractTwoTT(C_k,P_meas,2,2);
                %                     Skupdate = ContractTwoTT(Skupdate,C_k,2,2);
                %                     Skupdate = ContractTTtoTensor(Skupdate);
                %                     Skupdate = S_k_scalar + R;
                %
                %
                %                     TTMupdate.Cores{TTMupdate.NumCores} = TTMupdate.Cores{TTMupdate.NumCores}*(1/Skupdate);
                %                     UpdateTTM(:,:,k) = reshape(ContractTTtoTensor(TTMupdate),[n^d n^d]);
                %
                %                 end
                
                
                
                
                
            end
            
            
            TTKF_output = [m_meas, P_meas];
            %StabilityVecs =  [];%[Peig];
            StabilityVecs = [P_meas_squared_FrobNorm sum_alphas]; %[P_meas_squared_FrobNorm sum_alphas Peig];
            %VarianceAlphas = Variance;
            %             StabilityVecs.sum_alphas = sum_alphas;
            %             StabilityVecs.Peig = Peig;
            %             StabilityVecs.P_meas_squared_FrobNorm = P_meas_squared_FrobNorm;
            %             StabilityVecs.UpdateTTM = UpdateTTM;
        end
        
        
        
        
        %%
        function ConvergenceStatus = ConvergenceCheck(ConvCond,sum_alphas,P_meas_squared_FrobNorm,k)
            
            %%%%
            % • ConvergenceStatus = 0 : not converged!
            % • ConvergenceStatus = 1 : possibly converged! -> then we should
            % test the regressor, right?
            
            ConvergenceStatus = 0;  % unless if statements accepted
            
            if P_meas_squared_FrobNorm(k) <= 0.00001
                ConvergenceStatus = 1;
            end
            
            %k+1 because first index is initial (P0)
            if (P_meas_squared_FrobNorm(k)/P_meas_squared_FrobNorm(1) < ConvCond.FactorRemainingUncertainty) && (sum_alphas <  ConvCond.SumAlphaBound) && ConvergenceStatus ~= 1
                
                if k>(ConvCond.FactorMinUncertaintChangeNumIter+1)  %only after (ConvCond.FactorMinUncertaintChangeNumIter+1)
                    
                    %%% Condition 3: Norm difference per iteration
                    if P_meas_squared_FrobNorm(k) ~= 0 && k>(ConvCond.FactorMinUncertaintChangeNumIter+1)
                        ChangeNorm = P_meas_squared_FrobNorm((k):-1:(k-ConvCond.FactorMinUncertaintChangeNumIter))./P_meas_squared_FrobNorm(k-1:-1:(k-1-ConvCond.FactorMinUncertaintChangeNumIter));
                        PercChangeNormIter = 1-ChangeNorm
                    end
                    % • if all change in percentages > 0 -> (more certainty)
                    % stable. • if all change in percentges < bound, possibly
                    % converged.
                    if (all(PercChangeNormIter>0)) && (all(PercChangeNormIter<ConvCond.FactorMinUncertaintChangeIter))
                        ConvergenceStatus = 1;
                    end
                end
                
                
                %elseif P_meas_FrobNorm(k+1)/P_meas_FrobNorm(1)>1
                %error('Check ConvCond FactorRemainingUncertainty')
                
            end
            
            
            %P_meas_squared_FrobNorm(k)/P_meas_squared_FrobNorm(k-1);
            
            
            if (P_meas_squared_FrobNorm(k)/P_meas_squared_FrobNorm(k-1) > 1)
                warning('P-norm increased: less confidence in estimate!')
            end
        end
        
        
        
    end
end
