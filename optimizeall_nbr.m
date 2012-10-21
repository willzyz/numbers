function stackfull = optimizeall_nbr(pretrain_params, netconfig, data)

options.DerivativeCheck = false;
options.Method = 'lbfgs';
options.maxIter = 1000;
options.display = 'on';
options.logfile = 'log.txt'; % optional

fprintf(' ------- optimize deep autoencoder with L-BFGS ------- \n', i);
if netconfig.use_denoise
    [optall_params, obj_val, opt_output] = minFunc(@(p)autoeloss_denoise(p, ...
                                                      netconfig, ...
                                                      addnoise(data, ...
                                                      netconfig ...
                                                      .noise_level, 'binary'), data), pretrain_params, options);
else
    [optall_params, obj_val, opt_output] = minFunc(@(p)autoeloss(p, ...
                                                      netconfig, data), pretrain_params, options);
end

savename = sprintf('/afs/cs/u/wzou/scratch/numbers/savemodels/optimize_stackfull_exp%d.mat', netconfig.exp_count);
save(savename, 'optall_params', 'netconfig', 'obj_val', 'opt_output', '-v7.3');
