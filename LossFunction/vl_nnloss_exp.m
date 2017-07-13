function Y = vl_nnloss(X,c,dzdy,varargin)

% % pixel-level L2 loss
% % --------------------------------------------------------------------
% X = c(:,:,4:6,:) - X;
% c = c(:,:,1:3,:);
% if nargin <= 2 || isempty(dzdy)
% %     t = ((X-c).^2)./(2*c.^2 + 0.01) + ((X-c).^2)*8;
%     t = ((X-c).^2)./(2*c.^2 + 0.01);
%     Y = sum(t(:))/(size(X,4) * size(X,1) * size(X,2) * size(X,3) ); % reconstruction error per sample;
% else
% %     Y = bsxfun(@minus,X,c).*dzdy;   %backward funnction, 由faward对X求导得来
%     Y = -(bsxfun(@minus,X,c)).*dzdy*2./(2*c.^2 + 0.01);   %backward funnction, 由faward对X求导得来
% end
% pixel-level L2 loss
% % % --------------------------------------------------------------------

X = c(:,:,4:6,:) - X;
c = c(:,:,1:3,:);
if nargin <= 2 || isempty(dzdy)
%     t = ((X-c).^2)./(2*c.^2 + 0.01) + ((X-c).^2)*8;
    t = ((X-c).^2);
    Y = sum(t(:))/(size(X,4) * size(X,1) * size(X,2) * size(X,3) ); % reconstruction error per sample;
else
%     Y = bsxfun(@minus,X,c).*dzdy;   %backward funnction, 由faward对X求导得来
    Y = -(bsxfun(@minus,X,c)).*dzdy*2;   %backward funnction, 由faward对X求导得来
end

% function Y = vl_nnloss(X,c,dzdy,varargin)
% 
% % --------------------------------------------------------------------
% % pixel-level L2 loss
% % --------------------------------------------------------------------
% X1 = c(:,:,7:9,:) - X;
% X1(X1>=1) = 0.99;X1(X1<= -1) = -0.99;
% X2 = atanh(X1);
% % X2 = log(2/(1-X1)-1)/2;
% c = c(:,:,1:3,:);
% if nargin <= 2 || isempty(dzdy)
% %     t = ((X-c).^2)./(2*c.^2 + 0.01) + ((X-c).^2)*8;
%     t = ((X2-c).^2)./(2*c.^2 + 0.01);
%     Y = sum(t(:))/(size(X,4) * size(X,1) * size(X,2) * size(X,3) ); % reconstruction error per sample;
% else
% %     Y = bsxfun(@minus,X,c).*dzdy;   %backward funnction, 由faward对X求导得来
%     Y = (bsxfun(@minus,X2,c)).*(-dzdy*2).*(1/( 1-X1.^2 ) )   ./(2*c.^2 + 0.01);   %backward funnction, 由faward对X求导得来
% end


% function Y = vl_nnloss(X,c,dzdy,varargin)
% 
% % --------------------------------------------------------------------
% % pixel-level L2 loss
% % --------------------------------------------------------------------
% X = c(:,:,4:6,:)+ X.*c(:,:,7:9,:);
% d = c(:,:,7:9,:);
% c = c(:,:,1:3,:);
% if nargin <= 2 || isempty(dzdy)
% %     t = ((X-c).^2)./(2*c.^2 + 0.01) + ((X-c).^2)*8;
%     t = ((X-c).^2)./(2*c.^2 + 0.01);
%     Y = sum(t(:))/(size(X,4) * size(X,1) * size(X,2) * size(X,3) ); % reconstruction error per sample;
% else
% %     Y = bsxfun(@minus,X,c).*dzdy;   %backward funnction, 由faward对X求导得来
%     Y = bsxfun(@minus,X,c).*dzdy*2.*d ./ (2*c.^2 + 0.01);   %backward funnction, 由faward对X求导得来
% end
