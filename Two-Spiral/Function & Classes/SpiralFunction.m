function [x,y] = SpiralFunction(radius,mesh,n,d,initial_phase)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


x = zeros(0.5*(n^d),1);
y = zeros(0.5*(n^d),1);

for i=1:0.5*n^d
    phi     = (i/mesh)*pi + initial_phase
    r       = radius*((0.5*n^d)+20-i)/((0.5*n^d)+20);
    x(i)    = r*cos(phi);
    y(i)    = r*sin(phi);
end

end

