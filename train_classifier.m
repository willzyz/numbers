ref_nbrs = [8, 16]; 
colors = {'b', 'g', 'm', 'c'};
weightcost = 0;
count = 1;
figure;

for reference_nbr = ref_nbrs %[1e-6, 1e-7, 1e-10, 1e-20, 1e-30] %[0.1, 0.01, 0.001, 0.0001, 0.00001] %
    
    params_filename = sprintf('/afs/cs/u/wzou/scratch/numbers/savemodels/optimize_stackfull_exp%d.mat', netconfig.exp_count);
    load(params_filename);
    load zorzi_data.mat
    %D = bsxfun(@minus, D, netconfig.meanD);
    %D = bsxfun(@rdivide, D, netconfig.stdD);
    if netconfig.use_whiten        
        D = D - netconfig.meanD; 
        D = netconfig.whitenM*D; 
    end
    
    stack = params2stacksimple(optall_params, netconfig);    
    
    [~, h] = fwact(D, stack, @sigmoid_act, 0, 'autoe');
    h2 = h{2};
        
    addpath /afs/cs/u/wzou/scratch/classifiers/softmax/
    
    opt_params = train_sm(h2, (l>reference_nbr)+1, weightcost);
    
    z = softmax_func(opt_params*h2); 
    [~, idx] = max(z, [], 1); 
    fprintf('training accuracy %f\n', sum(idx' == ((l>reference_nbr) + 1 ))/length(idx));
    
    % ---- test on validation set ---- 
    load zorzi_valdata.mat
    
    if netconfig.use_whiten
        Dval = Dval - netconfig.meanD; 
        Dval = netconfig.whitenM*Dval; 
    end
    
    [~, hval] = fwact(Dval, stack, @sigmoid_act, 0, 'autoe');
    h2val = hval{2};
    
    z = softmax_func(opt_params*h2val);
    
    [~, idx] = max(z, [], 1); 
    
    % ---- validation accuracy ----
    fprintf('validation accuracy %f\n', sum(idx' == ((lval>reference_nbr) + 1 ))/length(idx));
    
    p = z(2, :);            
    lr_set{count} = []; 
    ave_prob_set{count} = [];
    for nbr = 1:32
        filter = (nbr == lval);
        lr = log(nbr)/log(reference_nbr);
        ave_prop = mean(p(filter));
        lr_set{count} = [lr_set{count}, lr];
        ave_prob_set{count} = [ave_prob_set{count}, ave_prop];
    end 
    plot(lr_set{count}, ave_prob_set{count}, ['o' colors{count}]); hold on; 
    count = count + 1;
end

l = [lr_set{1}, lr_set{2}];
a = [ave_prob_set{1}, ave_prob_set{2}];

custom_erf = @(w, x)(0.5*(1+erf((x-1)/(sqrt(2)*w))));
wopt = lsqcurvefit(custom_erf, 0.5, l, a);

% w = 0.8;
t = 1:0.01:6.1;
x = log(t);
y = custom_erf(wopt, x);
plot(x, y, 'r'); 
grid on
legend('ref. no. 8', 'ref. no. 16', 'sigmoid fit');
xlabel('Numerical ratio (log scale)');
ylabel('Average positive classifier coefficient');
title(['weber fraction ' num2str(wopt)]); 
