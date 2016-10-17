clear all, close all, clc

covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.2; hyp.lik = log(sn);

n = 30;
x = gpml_randn(0.3, n, 1);
K = feval(covfunc{:}, hyp.cov, x);
y = chol(K)'*gpml_randn(0.15, n, 1) + exp(hyp.lik)*gpml_randn(0.2, n, 1);
nu = 10; u = linspace(-1.3,1.3,nu)';
z = linspace(-3, 3, 201)';

%% FITC -- Carl's new implementation
hyp_cell.l = log(ell);
hyp_cell.s = log(sf);
hyp_cell.n = log(sn);
hyp_cell.m = 0;
hyp_cell.b = 0;
p.hyp = hyp_cell;
p.induce = u;
style.name = 'fitc';
style.linear_mean = 0;
p_opt = minimize(p, @sgp, -1000, x, y, style);
[m, s2] = sgp(p_opt, x, y, style, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(u)), 'xr')
plot(p_opt.induce, zeros(size(u)), 'ok')
title('FITC')

%% VFE -- Carl's new implementation
hyp_cell.l = log(ell);
hyp_cell.s = log(sf);
hyp_cell.n = log(sn);
hyp_cell.m = 0;
hyp_cell.b = 0;
p.hyp = hyp_cell;
p.induce = u;
style.name = 'vfe';
style.linear_mean = 0;
p_opt = minimize(p, @sgp, -1000, x, y, style);
[m, s2] = sgp(p_opt, x, y, style, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(u)), 'xr')
plot(p_opt.induce, zeros(size(u)), 'ok')
title('VFE')

%% new PEP following Carl's PEP/FITC implementation
% testing small alpha, this should be identical to VFE above
alpha = 0.001;
hyp_cell.l = log(ell);
hyp_cell.s = log(sf);
hyp_cell.n = log(sn);
hyp_cell.m = 0;
hyp_cell.b = 0;
p.hyp = hyp_cell;
p.induce = u;
style.name = 'pep'; style.alpha = alpha;
style.linear_mean = 0;
p_opt = minimize(p, @sgp, -1000, x, y, style);
[m, s2] = sgp(p_opt, x, y, style, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(u)), 'xr')
plot(p_opt.induce, zeros(size(u)), 'ok')
title(['PEP alpha=' num2str(alpha)])

%% new PEP following Carl's PEP/FITC implementation
alpha = 0.5;
hyp_cell.l = log(ell);
hyp_cell.s = log(sf);
hyp_cell.n = log(sn);
hyp_cell.m = 0;
hyp_cell.b = 0;
p.hyp = hyp_cell;
p.induce = u;
style.name = 'pep'; style.alpha = alpha;
style.linear_mean = 0;
p_opt = minimize(p, @sgp, -1000, x, y, style);
[m, s2] = sgp(p_opt, x, y, style, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(u)), 'xr')
plot(p_opt.induce, zeros(size(u)), 'ok')
title(['PEP alpha=' num2str(alpha)])

