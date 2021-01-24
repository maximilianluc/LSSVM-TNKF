clear all
close all
profile off
profile on
%gammas = [0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000];
%gammas = [0.004:0.001:0.01];  %0.004 and 0.007 is quite good
%gammas  = 0.007; %best wel goed met 0.05
%sigmas = [2.32];
%sigmas = [0.0005 0.005 0.05 0.5 5 50];  %0.05 en 0.5 het beste
%sigmas = [0.05:0.025:0.75] % 0.05 best wel goed
%sigmas = [0.025]
%gammas = [0.004:0.001:0.007]
%sigmas = [0.015:0.01:0.045]

%[gam=0.009 & sig2 = 0.025] was best goed - 0.0151


%gammas = [0.005 0.01 0.05 0.1 0.5 1 5 10]
%sigmas = [0.005:0.005:0.025]
%[0.05&0.01]

%gammas=[0.05:0.005:0.1]
%sigmas=0.009


%gammas = [0.0850]
%sigmas = [0.005:0.001:0.012]

gammas=0.0850;
sigmas=0.009;


perf = zeros(length(gammas),length(sigmas));

for k = 1:1:length(gammas)
    for q = 1:1:length(sigmas)
       gam = gammas(k);
       sig2= sigmas(q);
       run('TNKF_Silverbox.m')
       perf(k,q) = RMSE_validation; 
       clearvars -except perf k q gammas sigmas;
       if perf(k,q)< 10^(-4)
            break
       end
    end
end  



% Trunc_Par.RankTrunc_m           = inf; 
% Trunc_Par.Eps_m                 = 0.1;
% Trunc_Par.RankTrunc_P           = inf;
% Trunc_Par.Eps_P                 = 0.9;
% Trunc_Par.RankTrunc_C           = inf;
% Trunc_Par.Eps_C                 = 0.75;
% Trunc_Par.RankTrunc_S_k         = inf;
% Trunc_Par.Eps_S_k               = 0.25;
% Trunc_Par.RankTrunc_K_k         = inf;
% Trunc_Par.Eps_K_k               = 0.25;

% for R_m = [1:20]
%     for R_p = [1:20]
%         for R_c = [1:20]
%             for R_s = [2:20] %for 1 met r_k 1:20 al gedaan
%                 for R_k = [1:20]
%                     
%                     Trunc_Par.RankTrunc_m           = R_m; 
%                     Trunc_Par.Eps_m                 = 0;
%                     Trunc_Par.RankTrunc_P           = R_p;
%                     Trunc_Par.Eps_P                 = 0;
%                     Trunc_Par.RankTrunc_C           = R_c;
%                     Trunc_Par.Eps_C                 = 0;
%                     Trunc_Par.RankTrunc_S_k         = R_s;
%                     Trunc_Par.Eps_S_k               = 0;
%                     Trunc_Par.RankTrunc_K_k         = R_k;
%                     Trunc_Par.Eps_K_k               = 0;
%                     
%                     run('TNKF_Silverbox.m')
%                     
%                     perf(R_m,R_p,R_c,R_s,R_k) = RMSE_validation; 
%                     if perf(R_m,R_p,R_c,R_s,R_k)< 6*10^(-3)
%                          break
%                     end 
%                     
%                     clearvars -except perf R_m R_p R_c R_s R_k ;
%                     
%                 end
%             end
%         end
%     end
% end

                    
                    
                    





