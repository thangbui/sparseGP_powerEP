clear all, close all, clc

covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.2; hyp.lik = log(sn);

n = 30;
x = gpml_randn(0.3, n, 1);
K = feval(covfunc{:}, hyp.cov, x);
y = chol(K)'*gpml_randn(0.15, n, 1) + exp(hyp.lik)*gpml_randn(0.2, n, 1);
nu = 10; u = linspace(-1.3,1.3,nu)';
z = linspace(-3, 3, 201)';

%% FITC -- GPML
covfunc = @covSEiso; covfuncFITC = {@covFITC, {covfunc}, u};
hyp.xu = u;
hyp1 = minimize(hyp, @gp, -1000, @infFITC, [], covfuncFITC, likfunc, x, y);
[m s2] = gp(hyp1, @infFITC, [], covfuncFITC, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(hyp1.xu)), 'xr')
plot(hyp1.xu, zeros(size(hyp1.xu)), 'ok')
title('FITC - GPML')

%% FITC -- GPML
covfunc = @covSEiso; covfuncVFE = {@covVFE, {covfunc}, u};
hyp.xu = u;
hyp1 = minimize(hyp, @gp, -1000, @infVFE, [], covfuncVFE, likfunc, x, y);
[m s2] = gp(hyp1, @infVFE, [], covfuncVFE, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(hyp1.xu)), 'xr')
plot(hyp1.xu, zeros(size(hyp1.xu)), 'ok')
title('VFE - GPML')

%% PEP small alpha eg. VFE
alpha = 0.0001;
covfunc = @covSEiso; covfuncPEP = {@covPEP, {covfunc}, u, alpha};
hyp1 = minimize(hyp, @gp_new, -1000, @infPEP, [], covfuncPEP, likfunc, x, y);
[m s2] = gp_new(hyp1, @infPEP, [], covfuncPEP, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(hyp1.xu)), 'xr')
plot(hyp1.xu, zeros(size(hyp1.xu)), 'ok')
title('PEP(VFE) - GPML')

%% PEP - GPML
covfunc = @covSEiso; covfuncPEP = {@covPEP, {covfunc}, u, alpha};
hyp1 = minimize(hyp, @gp_new, -1000, @infPEP, [], covfuncPEP, likfunc, x, y);
[m s2] = gp_new(hyp1, @infPEP, [], covfuncPEP, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
figure()
plot(x, y, '+')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+b')
plot(u, zeros(size(hyp1.xu)), 'xr')
plot(hyp1.xu, zeros(size(hyp1.xu)), 'ok')
title('PEP GPML')
