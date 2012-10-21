function Xnoise = addnoise(X, perc, type)

% add noise to data
% X: input data
% perc: percentage noise of std deviation of the whole dataset
% this function adds gaussian noise according to mean and std of data

num_elements = numel(X);

if strcmp(type, 'gaussian')
    s = std(X(:));

    Xnoise = randn(size(X))*s*perc + X;
else
    Xnoise = X; 
    rndidx = randperm(num_elements); 
    num_noise = ceil(num_elements*perc);
    Xnoise(rndidx(1:num_noise)) = 0; 
    % setting randome elements of data to a fixed value
    % the model is then asked to fill in these data-points given context
end
