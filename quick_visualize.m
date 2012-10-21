for exp = 1000:1010
    fn = sprintf('devsave_exp%d_iter10000.mat', exp);
    load(fn);
    stack=params2stacksimple(params, netconfig);
    figure; pf(stack{1}.w'); title(['alpha1 ' num2str(netconfig.alpha1) ' beta1 ' num2str(netconfig.beta1)]);
end
