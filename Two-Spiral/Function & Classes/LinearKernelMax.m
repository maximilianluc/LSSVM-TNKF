function [RBF_row] = LinearKernelMax(Data,gamma,n,d,k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%Data2 = Data';

RBF_row = zeros(1,n^d);
diff = zeros(n^d,size(Data,2));
i = k;
diff(1:n^d,:) = Data(i,:)-Data;


%diff(1:n^d,:) = bsxfun(@minus, Data(i,:), Data); same as above
%for j = 1:1:n^d %j is the time 
    %diff(:,j) =    (Data(i,:)-Data(j,:))';  %instead of Data(:,i)-Data(:,j);
    %RBF_row(1,j) = exp( -(sum((diff(j,:).*diff(j,:))./(2*sig2))) ); %werkt =)
    
%end

RBF_row =  sum(diff.*diff,2)' ;
% for j = 1:1:n^d
%     %diff(:,j) =    (Data(i,:)-Data(j,:))';  %instead of Data(:,i)-Data(:,j);
%     RBF_row(1,j) = exp( -(sum((diff(j,:).*diff(j,:))./(2*sig2))) ); %werkt =)
% end

RBF_row(1,i) = RBF_row(1,i) + 1/gamma;
end

