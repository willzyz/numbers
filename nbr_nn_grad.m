function [f, g, fplain] = nbr_nn_grad(theta, numh1, numh2, numh3, X, y, wc_reg, sp_reg)

% one layer neural network prediction of the next time step
% w:    input parameters
% numH: number of hidden units

[w1, b1, w2, b2, wc, bc] = reorg_theta(theta, X, numh1, numh2, numh3);

%% fwprop
z1 = bsxfun(@plus, w1*X, b1);
a1 = sigmoid_act(z1);

z2 = bsxfun(@plus, w2*a1, b2);
a2 = sigmoid_act(z2);

zc = bsxfun(@plus, wc*a2, bc);
ac = sigmoid_act(zc);

fplain = 0.5*sum(sum((ac-y).^2))/size(X, 2)/size(y, 1); % mse cost function
f = fplain + 0.5*sum(theta.^2)*wc_reg;
f = f + 0.5*sum(sum((a1-(-0.9)).^2))/numel(a1)*sp_reg;
% f = f + 0.5*sum(sum((a2-(-0.9)).^2))/numel(a2)*sp_reg;

%% bkprop 
d = (ac-y)/size(X, 2)/size(y, 1); 
d = d.* sigmoid_derv(ac); % bkprop classifier sigmoid
% d = d.* (1-ac.^2); % bkprop classifier sigmoid
gbc = sum(d, 2); % classifier bias gradient
gwc = d * a2'; % classifier weights gradient

d = wc'*d; %bkprop through classifier weights

% d = d + (a2 - (-0.9))/numel(a2)*sp_reg;

d = d .* sigmoid_derv(a2); % bkprop n layer nonlin
gb2 = sum(d, 2); % n layer bias gradient
gw2 = d * a1'; % n layer weights gradient

d = w2'*d;          %bkprop through classifier weights

d = d + (a1 - (-0.9))/numel(a1)*sp_reg;
d = d .* sigmoid_derv(a1);    % bkprop n layer nonlin
gb1 = sum(d, 2);       % n layer bias gradient
gw1 = d * X';          % n layer weights gradient

g = [gw1(:); gb1(:); gw2(:); gb2(:); gwc(:); gbc(:)];

g = g + theta*wc_reg;
