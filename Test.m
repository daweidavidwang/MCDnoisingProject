%%% test the model performance
clear;clc;close;
matConvNetPath = 'E:\matconvnet-1.0-beta22\matconvnet-1.0-beta22\';
run( fullfile(matConvNetPath,'matlab','vl_setupnn.m') );
% clear; clc;
format compact;

addpath(fullfile('data','utilities'));
folderTest  = fullfile('E:\datasetcombine\32SPP\jpg\'); %%% test dataset
folderTestGt  = fullfile('E:\datasetcombine\GT\jpg\'); %%% test dataset
folderTestPre = fullfile('./','pre');
showResult  = 1;
useGPU      = 0;
pauseTime   = 0;

SSIMall= [];
PSNRall = [];
%%% read images
ext         =  {'*.jpg'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end


epoch       = 43;

%%% load Gaussian denoising model
%load(fullfile('data','model_MC_Res_Bnorm_Adam','kmeans-15layerWith1x1',['model_MC_Res_Bnorm_Adam','-epoch-',num2str(epoch),'.mat']));

load(fullfile('data','model_MC_Res_Bnorm_Adam',['model_MC_Res_Bnorm_Adam','-epoch-',num2str(epoch),'.mat']));
% load(fullfile('I:\DnCNN\TrainingCodes\RLCNN_v0.0\data\model_MC_Res_Bnorm_Adam_170320_correctBackward_allfeatureResidual',['model_MC_Res_Bnorm_Adam','-epoch-',num2str(epoch),'.mat']));

net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);
%%% move to gpu
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end
%%% PSNR and SSIM
% result.PSNRs = zeros(1,length(filePaths));
% result.SSIMs = zeros(1,length(filePaths));
% result.name = [];
time = zeros(length(filePaths),1);
for i = 1:length(filePaths)
        %%% read images
    tic;
    label = imread(fullfile(folderTestGt,[filePaths(i).name(1:end-12),'.jpg']  )) ;
%     label(label>1) = 1;
    
    input_jpg =  imread( fullfile(folderTest,[filePaths(i).name] ) );
%     input_exr(input_exr>1) = 1;
    
    input = load( fullfile(folderTest,['../Feature/',filePaths(i).name(1:end-12),'.mat'] ) );
    input = single(reshape(input.doublefeature,[size(input_jpg,2) size(input_jpg,1) 18]));
    input = permute( input(:,:,3:end), [2,1,3] );
    input(:,:,1:3) = im2single(input_jpg);
%     clear input_exr;

    
    for j = 4:16
        input(:,:,j) = ( input(:,:,j) - min(min(input(:,:,j))) )/( max(max(input(:,:,j))) - min(min(input(:,:,j))) );
    end

%     for j = 1:16
%         tmp = input(:,:,j);
%         input(:,:,j) = (input(:,:,j) - mean(tmp(:)))/std(tmp(:));
%         %         input_im(:,:,j) = ( input_im(:,:,j) - min(min(input_im(:,:,j))) )/( max(max(input_im(:,:,j))) - min(min(input_im(:,:,j))) );
%     end
%     


    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    %%% convert to GPU
    if useGPU
        input = gpuArray(input);
    end
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
%     output = res(end).x;
%     output = input_exr+ res(end).x ;
%     output = input_exr+res(end).x .*( cat(3,medfilt2(input_exr(:,:,1)),input_exr(:,:,2),input_exr(:,:,3) ) );
    output = input(:,:,1:3) - res(end).x;
    output(output>1) = 1;output(output<0) = 0;
   %%% convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    output = output;
    imwrite(im2uint8(output),fullfile(folderTestPre,[filePaths(i).name(1:end-4),'.png']),'png');
    time(i,1) = toc;
    
    
    PSNRCur = psnr(im2uint8(output),im2uint8(label));
    SSIMCur = ssim(im2uint8(output),im2uint8(label));
    result(i).PSNRs = PSNRCur;
    result(i).SSIMs = SSIMCur;
    result(i).name = filePaths(i).name;
    if showResult
        imshow(cat(2,(im2uint8(label)),(im2uint8(input_jpg)),(im2uint8(output)),(im2uint8(res(end).x))));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
        disp SSIM:;disp (SSIMCur);
        disp PSNR:;disp(PSNRCur);
        pause(pauseTime)
    end
end
    SSIMall(epoch,:) = [result.SSIMs];
    PSNRall(epoch,:) = [result.PSNRs];
disp([mean([result.PSNRs]),mean([result.SSIMs])]);


subplot(1,2,1);
hold on;
for i = 1:size(SSIMall,2)
    plot(SSIMall(:,i),'*');
end
subplot(1,2,2);
hold on;
for i = 1:size(PSNRall,2)
    plot(PSNRall(:,i),'+');
end
legend(filePaths.name)
% saveas ( gcf ,fullfile('data',modelName,'accuracy.png' ),'png');
% xlswrite(fullfile('data',modelName, 'accuracy.xls'),SSIMall,'SSIM');
% xlswrite( fullfile('data',modelName, 'accuracy.xls'),PSNRall,'PSNR');