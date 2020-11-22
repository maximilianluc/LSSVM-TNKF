function [RBF_row] = LinearKernelMax(Data,gamma,n,d,k)
RBF_row = zeros(1,n^d);
diff = zeros(n^d,size(Data,2));
i = k;
diff(1:n^d,:) = Data(i,:)-Data;
RBF_row =  sum(diff.*diff,2)' ;
RBF_row(1,i) = RBF_row(1,i) + 1/gamma;
end

