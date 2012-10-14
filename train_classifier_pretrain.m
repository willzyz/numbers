for weightcost = 0%[1e-6, 1e-7, 1e-10, 1e-20, 1e-30]%[0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    load second_test.mat
    load zorzi_data.mat
    D = reshape(D, [], size(D, 3)); 
    
    stack = params2stacksimple(opt_params, netconfig);
    second_stack = params2stacksimple(second_opt_params, second_netconfig);

    h1 = sigmoid_act(bsxfun(@plus, stack{1}.w*D, stack{1}.b));

    h2 = sigmoid_act(bsxfun(@plus, second_stack{1}.w*h1, second_stack{1}.b));

    addpath /afs/cs/u/wzou/scratch/classifiers/softmax/

    opt_params = train_sm(h2, (l>16)+1, weightcost, 0);

    % ---- test on validation set ---- 
    load zorzi_valdata.mat
    D = reshape(D, [], size(D, 3)); 

    h1 = sigmoid_act(bsxfun(@plus, stack{1}.w*D, stack{1}.b));

    h2 = sigmoid_act(bsxfun(@plus, second_stack{1}.w*h1, second_stack{1}.b));

    z = softmax_func(opt_params*h2);

    [~, idx] = max(z, [], 1); 

    %% accuracy
    weightcost
    sum(idx' == ((l>16) + 1 ))/length(idx)
end
