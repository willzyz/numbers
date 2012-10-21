close all

beta1 = alpha1*1000; 
% exp = 50; 
use_denoise = 0; 
use_whiten = 0; 
noise_level = 0;

% ---- define netconfig ----

hsz1 = 80; % default 80
hsz2 = 400; % default 400

netconfig.inputsize = 900;
netconfig.layersizes = {hsz1, hsz2, hsz1, 900};
netconfig.lambda = 0;
netconfig.act_func = @sigmoid_act;
netconfig.der_func = @sigmoid_deriv;
netconfig.cost_func = @spcost_logcosh;
netconfig.weightcost = 0;
netconfig.use_gpu = 0;
netconfig.maxIter_pretrain = 1000;
netconfig.maxIter_optimize = 1000;
netconfig.maxSGDIter = 5e6;
netconfig.saveInterval = 1e4;
netconfig.alpha1 = alpha1; 
netconfig.beta1 = beta1; 
netconfig.alpha2 = alpha1; 
netconfig.beta2 = beta1; 

netconfig.exp_count  = exp;
netconfig.use_denoise = use_denoise;
netconfig.noise_level = noise_level;
netconfig.use_whiten = use_whiten; 

fprintf('loading training data ......\n');
load zorzi_data.mat
%D = D(:, 1:5e3);
%netconfig.meanD = mean(D, 2);
%D = bsxfun(@minus, D, netconfig.meanD);
%netconfig.stdD = std(D, 1, 2); 
%D = bsxfun(@rdivide, D, netconfig.stdD);

if netconfig.use_whiten
    netconfig.meanD = mean(D(:));
    D = D - netconfig.meanD; 
    %netconfig.stdD = std(D(:));
    %D = D./netconfig.stdD;
    
    fprintf('whitening training data ......\n'); 
    [V, E, ~] = pca(D); 
    netconfig.whitenM = E*V; 
    D = netconfig.whitenM*D; 
end

% stackfull = pretrain_nbr(netconfig, D);
% pretrain_params = stack2paramssimple(stackfull, netconfig);
% optimizeall_nbr(pretrain_params, netconfig, D);

% train_classifier;

opt_params = simulate_development(netconfig, D);

%optimize_development(netconfig, D); 
