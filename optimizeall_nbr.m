function stackfull = optimizeall_nbr(pretrain_params, netconfig, data)

options.DerivativeCheck = false;
options.Method = 'lbfgs';
options.maxIter = 1000;
options.display = 'on';
options.logfile = 'log.txt'; % optional

[optall_params, obj_val, opt_output] = minFunc(@(p)autoeloss(p, netconfig, data), pretrain_params, options);

savename = sprintf('/afs/cs/u/wzou/scratch/numbers/savemodels/optimize_stackfull_exp%d.mat', netconfig.exp_count);
save(savename, 'optall_params', 'netconfig', 'obj_val', 'opt_output', '-v7.3');
