function optimize_development(netconfig, D, params)
maxSGDIter = netconfig.maxSGDIter; 
saveInterval = netconfig.saveInterval;
alpha = netconfig.alpha;
beta = netconfig.beta;

storelen = 1000;
fstore = [];
estfhistory = [];
iter = 1;
fprintf('running SGD development simulation......\n')
while iter<=maxSGDIter
    fprintf('%d', iter); 
    if mod(iter, 20)
        fprintf('\n');
    end
    % ----- choose random data sample -----
    Ds = D(:, randi([1, size(D, 2)])); 
    
    % ----- evaluate gradient -----    
    [f, g] = autoeloss(params, netconfig, Ds); 
    
    fstore = [fstore, f];
    if length(fstore)>storelen
        fstore = fstore(2:end);
    end
    estfhistory = [estfhistory, mean(fstore)];
    
    % take SGD step 
    mult = alpha/(beta + iter);
    params = params - mult*g;
        
    iter = iter + 1;
    subplot(2, 1, 2); plot(estfhistory); drawnow();
    if mod(iter, 30)
        stack = params2stacksimple(params, netconfig);
        subplot(2, 1, 1); pf(stack{1}.w');
    end
    % save parameters
    if mod(iter, saveInterval) == 0
        filename = sprintf('/afs/cs/u/wzou/scratch/numbers/development_savemodel/devsave_exp%d_iter%d.mat', netconfig.exp_count, iter);
        save(filename, 'params', 'netconfig', 'estfhistory');
        print('-depsc', [filename, '.eps']);
    end
end
