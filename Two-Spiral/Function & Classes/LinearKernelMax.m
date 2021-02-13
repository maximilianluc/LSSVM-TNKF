function [RBF_row] = LinearKernelMax(Data,gamma,n,d,k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%Data2 = Data';

RBF_row = zeros(1,n^d);
diff = zeros(n^d,size(Data,2));
i = k;
diff(1:n^d,:) = Data(i,:)-Data;

RBF_row =  sum(diff.*diff,2)' ;

RBF_row(1,i) = RBF_row(1,i) + 1/gamma;
end

