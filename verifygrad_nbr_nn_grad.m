addpath ~/scratch/gradcheck/
addpath ~/scratch/timeseries/nn/

insz = 20;
numc = 3;
numh1 = 5;
numh2 = 3;
numh3 = 3;
wc_reg = 1;
sp_reg = 1;

X = randn(insz*numc, 50); 
y = randn(numh3, 50); 

% t = formbhat(numh1*(insz-cinsz1+1), randn(numh1, 1));
% class_input = numel(t);

theta = randn(numc*insz*numh1+numh1+numh1*numh2+numh2+numh2*numh3+numh3, 1); 

checkgradfull('nbr_nn_grad', theta, 1e-7, numh1, numh2, numh3, X, y, wc_reg, sp_reg); 

% checkgradfull('nn_grad_conv', theta, 1e-7, cinsz1, numh1, cinsz2, numh2, class_input, numh3, X, ...
%               y, wc_reg, numc);
