function params = simulate_development(netconfig, D)
    maxSGDIter = netconfig.maxSGDIter; 
    saveInterval = netconfig.saveInterval;
    alpha1 = netconfig.alpha1;
    beta1 = netconfig.beta1;
    alpha2 = netconfig.alpha2;
    beta2 = netconfig.beta2;

    % initialize parameters
    % params = 0.01*randn(computeParamLen(netconfig), 1);
    nc = struct;
    nc.inputsize = size(D, 1);
    nc.layersizes = { netconfig.layersizes{1} nc.inputsize };
    l1params = 0.01*randn(computeParamLen(nc), 1); 
    nc = struct;
    nc.inputsize = netconfig.layersizes{1};
    nc.layersizes = { netconfig.layersizes{2} nc.inputsize };
    l2params = 0.01*randn(computeParamLen(nc), 1);

    fstore1 = [];
    fstore2 = [];

    estfhistory1 = [];
    estfhistory2 = [];
    storelen = 1000;

    iter = 1;
    fprintf('running SGD development simulation......\n')
    while iter<=maxSGDIter
        fprintf('%d', iter); 
        if mod(iter, 20)
            fprintf('\n');
        end
        % ----- choose random data sample -----
        Ds = D(:, randi([1, size(D, 2)]));

        % ----- evaluate gradient layer 1 -----
        nc = struct;
        nc.inputsize = size(D, 1);
        nc.layersizes = { netconfig.layersizes{1} nc.inputsize };     
        nc.act_func = netconfig.act_func;%@tanh_act;
        nc.der_func = netconfig.der_func;%@tanh_deriv;    
        nc.tieweights = 1;
        nc.use_gpu = netconfig.use_gpu;
        nc.cost_func = netconfig.cost_func;

        [f1, g] = autoeloss(l1params, nc, Ds);

        % take SGD step layer 1
        mult = alpha1/(beta1 + iter);
        l1params = l1params - mult*g;

        % forward prop layer 1 using current params
        stack = params2stacksimple(l1params, nc);
        [~, h] = fwact(Ds, stack, @sigmoid_act, 0, 'autoe');
        d = h{1};
        
        fstore1 = [fstore1, f1];
        if length(fstore1)>storelen
            fstore1 = fstore1(2:end);
        end
        estfhistory1 = [estfhistory1, mean(fstore1)];
        
        if 1
        % evaluate gradient layer 2
        nc = struct;
        nc.inputsize = netconfig.layersizes{1};
        nc.layersizes = { netconfig.layersizes{2} nc.inputsize };     
        nc.act_func = netconfig.act_func;%@tanh_act;
        nc.der_func = netconfig.der_func;%@tanh_deriv;    
        nc.tieweights = 1;
        nc.use_gpu = netconfig.use_gpu;
        nc.cost_func = netconfig.cost_func;

        [f2, g] = autoeloss(l2params, nc, d);

        % take SGD step layer 2
        mult = alpha2/(beta2 + iter);
        l2params = l2params - mult*g;

        params = [l1params; l2params];
        
        fstore2 = [fstore2, f2];
        if length(fstore2)>storelen
            fstore2 = fstore2(2:end);
        end
        estfhistory2 = [estfhistory2, mean(fstore2)];
        end
        
        subplot(2, 1, 2); plot(estfhistory1, 'b'); hold on; 
        title(['obj : ', num2str(estfhistory1(end))]);
        subplot(2, 1, 2); plot(estfhistory2, 'r'); 
        if mod(iter, 30)
            stack = params2stacksimple(params, netconfig);
            %stack = params2stacksimple(l1params, nc);
            subplot(2, 1, 1); pf(stack{1}.w');
        end
        drawnow();
        % save parameters
        if mod(iter, saveInterval) == 0
            filename = sprintf('/afs/cs/u/wzou/scratch/numbers/development_savemodel/devsave_exp%d_iter%d.mat', netconfig.exp_count, iter);
            save(filename, 'params', 'netconfig', 'estfhistory1', 'estfhistory2');
            % print('-depsc', [filename, '.eps']);
        end
        iter = iter + 1;
    end
