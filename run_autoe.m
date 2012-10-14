addpath /afs/cs/u/wzou/scratch/autoencoder/

load zorzi_data.mat

D = reshape(D, [], size(D, 3));

inputsize = size(D, 1);
hiddensize = 80;

netconfig.inputsize = inputsize;
netconfig.layersizes = {hiddensize, inputsize};
netconfig.lambda = 0;
netconfig.act_func = @sigmoid_act;
netconfig.der_func = @sigmoid_deriv;
netconfig.cost_func = @spcost_logcosh;
netconfig.weightcost = 0;
netconfig.use_gpu = 0;

params = 0.01*randn(computeplen(netconfig), 1);

options.DerivativeCheck = false;
options.Method = 'lbfgs';
options.maxIter = 10000;
options.display = 'on';
options.logfile = 'log.txt'; % optional

opt_params = minFunc(@(p) autoeloss(p, netconfig, D), params, options);

save second_test.mat opt_params netconfig
