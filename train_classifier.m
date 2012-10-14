for weightcost = [0] %[1e-6, 1e-7, 1e-10, 1e-20, 1e-30] %[0.1, 0.01, 0.001, 0.0001, 0.00001] %
    
    params_filename = sprintf('/afs/cs/u/wzou/scratch/numbers/savemodels/optimize_stackfull_exp%d.mat', netconfig.exp_count);
    load(params_filename);
    load zorzi_data.mat
    
    stack = params2stacksimple(optall_params, netconfig);    
    
    [~, h] = fwact(D, stack, @sigmoid_act, 0, 'autoe');
    h2 = h{2};
        
    addpath /afs/cs/u/wzou/scratch/classifiers/softmax/
    
    opt_params = train_sm(h2, (l>16)+1, weightcost);
    
    z = softmax_func(opt_params*h2); 
    [~, idx] = max(z, [], 1); 
    fprintf('training accuracy %f\n', sum(idx' == ((l>16) + 1 ))/length(idx));
    
    % ---- test on validation set ---- 
    load zorzi_valdata.mat
    
    [~, hval] = fwact(Dval, stack, @sigmoid_act, 0, 'autoe');
    h2val = hval{2};
    
    z = softmax_func(opt_params*h2val);
    
    [~, idx] = max(z, [], 1); 
    
    %% validation accuracy
    weightcost
    fprintf('validation accuracy %f\n', sum(idx' == ((lval>16) + 1 ))/length(idx));
    
end
