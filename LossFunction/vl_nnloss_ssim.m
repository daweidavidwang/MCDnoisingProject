function Y = vl_nnloss(X,c,dzdy,varargin)
%SSIM-loss function from PL4NN for matconvnet 
X = c(:,:,4:6,:) - X;%output
c = c(:,:,1:3,:);%label
[height,width,channel,batch] = size(c);
sigma = 5.;
w = -(width/2):1:((width/2)-1);
w = w.^2;
w = exp((-1.*w)./(2*sigma^2));
w = w'*w;
w = w./sum(w(:));
w = reshape(w,width,width,1,1);
w = repmat(w,1,1,3,batch);
c1 = 0.01^2;
c2 = 0.03^2;
ux = sum(w.*c,2);
ux = sum(ux,1);
uy = sum(w.*X,2);
uy = sum(uy,1);
sigmax2 = sum(sum(w.*(c.^2),2),1)-ux.^2;
sigmay2 = sum(sum(w.*(X.^2),2),1)-uy.^2; 
sigmaxy = sum(sum(w.*(X.*c),2),1)-ux.*uy;
lp = ((ux.*uy.*2)+c1)./(ux.^2+uy.^2+c1);
csp = ((sigmaxy.*2)+c2)./(sigmax2+sigmay2+c2);
ux = repmat(ux,height,width,1,1);
uy = repmat(uy,height,width,1,1);
sigmax2 = repmat(sigmax2,height,width,1,1);
sigmay2 = repmat(sigmay2,height,width,1,1);
sigmaxy = repmat(sigmaxy,height,width,1,1);
if nargin <= 2 || isempty(dzdy)
    tttt  = 1-lp.*csp;
    Y = sum(tttt(:))./(channel*batch);
else
    csp = repmat(csp,height,width,1,1);
    lp = repmat(lp,height,width,1,1);
    dl = (2.*w.*(uy-ux.*lp)./(ux.^2 + uy.^2+c1));
    dcs = (2./(sigmax2+sigmay2+c2)).*((c-ux)-csp.*(X-uy)).*w;
    Y = -(dl.*csp + lp.*dcs)*dzdy/(channel*batch);

end

%%L1 loss part
