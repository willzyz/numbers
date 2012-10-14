addpath /afs/cs/u/wzou/scratch/autoencoder/

load zorzi_data.mat
load second_test.mat

D = reshape(D, [], size(D, 3));

stack = params2stacksimple(opt_params, netconfig); 

h = sigmoid_act(bsxfun(@plus, stack{1}.w*D, stack{1}.b));

% ---- train second layer ----
inputsize = size(h, 1);
hiddensize = 400;

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
options.maxIter = 1500;
options.display = 'on';
options.logfile = 'log.txt'; % optional

second_opt_params = minFunc(@(p) autoeloss(p, netconfig, h), params, options);

second_netconfig = netconfig;

save second_test.mat second_opt_params second_netconfig -append
