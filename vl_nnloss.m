function Y = vl_nnloss(X,c,dzdy,varargin)
%%L1
X = c(:,:,4:6,:) - X;
c = c(:,:,1:3,:);
if nargin <= 2 || isempty(dzdy)
    t = abs(X-c);
    Y = sum(t(:))/(size(X,4) * size(X,1) * size(X,2) * size(X,3) ); % reconstruction error per sample;
else
    Y = -sign((bsxfun(@minus,X,c))).*dzdy;   %backward funnction, 由faward对X求导得来

end

