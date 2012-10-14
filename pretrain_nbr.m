function stackfull = pretrain_nbr(netconfig, data)

% ---- pretrain function for a deep auto-encoder ----

stackfull = cell(numel(netconfig.layersizes), 1); 
nlayers = numel(netconfig.layersizes); 
obj_vals = cell(nlayers/2, 1); 
opt_outputs = cell(nlayers/2, 1);

dim_prev_layer = netconfig.inputsize; 

for i = 1:nlayers/2
    nc = struct;
    nc.inputsize = dim_prev_layer;
    nc.layersizes = { netconfig.layersizes{i} dim_prev_layer }; dim_prev_layer = netconfig.layersizes{i};
    
    nc.act_func = netconfig.act_func;%@tanh_act;
    nc.der_func = netconfig.der_func;%@tanh_deriv;
    
    % nc.smooth_size = 2;
    % nc.targetact = 0.001;
    % nc.lambda = 10;
    nc.gamma = 0;
    nc.tieweights = 1;
    nc.use_gpu = netconfig.use_gpu;
    nc.cost_func = netconfig.cost_func;
    
    plen   = computeParamLen(nc);
    params = randn(plen, 1)*0.1;
    
    if netconfig.use_gpu
        params = gsingle(params); 
    end
    
    % add noise to data
    % data_noise = addnoise(data, 0); 
    
    % run minFunc
    options.DerivativeCheck = false;
    options.Method = 'lbfgs';
    options.maxIter = 1000;
    options.display = 'on';
    options.logfile = 'log.txt'; % optional
    
    fprintf('training layer %d\n', i);
    [optparams, value, output] = minFunc( @(p) autoeloss(p, nc, data), ...
                                params, options);
    obj_vals{i} = value;
    opt_outputs{i} = output;
    
    % forward propagate data
    stack = params2stacksimple(optparams, nc);
    data = nc.act_func(bsxfun(@plus, stack{1}.w*data, stack{1}.b));
    
    % update the full stack
    stackfull{i}.w = double(stack{1}.w);
    stackfull{i}.b = double(stack{1}.b);
    stackfull{nlayers-i+1}.w = double(stack{2}.w);
    stackfull{nlayers-i+1}.b = double(stack{2}.b);
    
    savename = sprintf('/afs/cs/u/wzou/scratch/numbers/savemodels/pretrain_stackfull_exp%d.mat', netconfig.exp_count);
    save(savename, 'stackfull', 'netconfig', 'obj_vals', 'opt_outputs', '-v7.3');
end
