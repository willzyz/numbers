exp = 3;

% ---- define netconfig ----

hsz1 = 200; % default 80
hsz2 = 400; % default 400
hsz3 = 600;

netconfig.inputsize = 900;
netconfig.layersizes = {hsz1, hsz2, hsz3, hsz2, hsz1, 900};
netconfig.lambda = 0;
netconfig.act_func = @sigmoid_act;
netconfig.der_func = @sigmoid_deriv;
netconfig.cost_func = @spcost_logcosh;
netconfig.weightcost = 0;
netconfig.use_gpu = 0;
netconfig.exp_count  = exp;

load zorzi_data.mat

stackfull = pretrain_nbr(netconfig, D);
% load /afs/cs/u/wzou/scratch/numbers/savemodels/pretrain_stackfull_exp1.mat; 
pretrain_params = stack2paramssimple(stackfull, netconfig);
optimizeall_nbr(pretrain_params, netconfig, D);

