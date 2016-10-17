function [nlml, dnlml] = sgp(p, inputs, target, style, test)

ridge = 1e-06;               % relative jitter to make matrix better conditioned
linear_mean = style.linear_mean;
switch style.name, 
    case 'fitc', fitc = 1; vfe = 0; pep=0; alpha=1;
    case 'vfe', vfe = 1; fitc = 0; pep=0; alpha=1;
    case 'pep', vfe = 0; fitc = 0; pep=1; alpha=style.alpha;
end
induce = p.induce; hyp = p.hyp;                             % shorthand notation
[N, D] = size(inputs); M = size(induce,1); nlml = 0; dnlml.induce = zeros(M, D);

for e = 1:length(hyp)
  y = target(:,e) + linear_mean * (-inputs*hyp(e).m - hyp(e).b);
  l = exp(hyp(e).l); s2 = exp(2*hyp(e).s); n2 = exp(2*hyp(e).n);
  u = bsxfun(@rdivide, induce, l');                     % scaled inducing inputs
  x = bsxfun(@rdivide, inputs, l');                     % scaled training inputs
  Kuu = s2*(exp(-maha(u,u)/2) + ridge*eye(M));
  Kuf = s2*exp(-maha(u,x)/2);
  L = chol(Kuu)';
  V = L\Kuf;
  r = s2 - sum(V.*V,1)';                % diagonal residual Kff - Kfu Kuu^-1 Kuf
  G = fitc*r + pep*alpha*r + n2; iG = 1./G;
  A = eye(M) + V*bsxfun(@times,iG,V');
  J = chol(A)';
  B = J\V;
  z = iG.*y - (y'.*iG'*B'*B.*iG')';
  nlml = nlml + y'*z/2 + sum(log(diag(J))) + sum(log(G))/2 ...
    + vfe*sum(r)/n2/2 + pep*(1-alpha)/alpha*sum(log(1+alpha*r/n2))/2 ...
    + N*log(2*pi)/2;
  if nargin == 5                                              % make predictions
    beta = (y'.*iG'*B'/J/L)';
    W = L'\(eye(M)-eye(M)/J'/J)/L;
    Ktu = s2*exp(-maha(bsxfun(@rdivide,test,l'), u)/2);
    nlml = Ktu*beta + linear_mean * (test*hyp.m + hyp.b);
    dnlml = s2+n2-sum(Ktu*W.*Ktu,2);
  elseif nargout == 2                                      % compute derivatives
    R = L'\V;
    RiG = bsxfun(@times,R,iG');
    RdQ = -R*z*z' + RiG - bsxfun(@times,RiG*B'*B,iG');
    dG = z.^2 - iG + iG.^2.*sum(B.*B,1)';
    RdQ2 = RdQ + bsxfun(@times, R, fitc*dG' + pep*alpha*dG' - vfe/n2 ...
        - pep*(1-alpha)./(1+alpha*r'/n2)/n2);
    KW = Kuf.*RdQ2;
    KWR = Kuu.*(RdQ2*R');
    P = KW*x + bsxfun(@times, sum(KWR, 2) - sum(KW, 2), u) - KWR*u;
    dnlml.induce = dnlml.induce + bsxfun(@rdivide, P, l');
    dnlml.hyp(e).l = -sum(P.*u,1) ...
                            - sum((KW'*u - bsxfun(@times, sum(KW',2), x)).*x,1);
    dnlml.hyp(e).n = -sum(dG)*n2 - vfe*sum(r)/n2 ...
        - pep*(1-alpha)*sum(r./(1+alpha*r/n2))/n2;
    dnlml.hyp(e).s = sum(sum(Kuf.*RdQ)) - fitc*r'*dG - pep*alpha*r'*dG ...
        + vfe*sum(r)/n2 + pep*(1-alpha)*sum(r./(1+alpha*r/n2))/n2;
    dnlml.hyp(e).b = linear_mean * -sum(z);
    dnlml.hyp(e).m = linear_mean * -inputs'*z;
  end
end
