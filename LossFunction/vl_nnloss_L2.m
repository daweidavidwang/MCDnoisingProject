function Y = vl_nnloss(X,c,dzdy,varargin)
%%L2 loss
X = c(:,:,4:6,:) - X;
c = c(:,:,1:3,:);
if nargin <= 2 || isempty(dzdy)
    t = ((X-c).^2);
    Y = sum(t(:))/(size(X,4) * size(X,1) * size(X,2) * size(X,3) ); % reconstruction error per sample;
else
    Y = -(bsxfun(@minus,X,c)).*dzdy*2;   %backward funnction, 由faward对X求导得来
end

