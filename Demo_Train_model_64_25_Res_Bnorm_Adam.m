%%% training data (clean images) first.
clear;clc;close all;
dbstop if error;
matConvNetPath = 'E:\matconvnet-1.0-beta22\matconvnet-1.0-beta22\';
run( fullfile(matConvNetPath,'matlab','vl_setupnn.m') );

rng('default')

%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------
opts.modelName        = 'model_MC_Res_Bnorm_Adam'; %%% model name
% opts.learningRate     = [logspace(-3,-3,30) logspace(-4,-4,20)];%%% you can change the learning rate
opts.learningRate     = [logspace(-3,-3,20) logspace(-4,-4,15) logspace(-3,-3,25) logspace(-4,-4,15) logspace(-4,-4,15) logspace(-4,-4,15)];%%% you can change the learning rate
% opts.learningRate     = [logspace(-4,-4,30)];


% opts.learningRate     = [logspace(-3,-3,5)];%%% you can change the learning rate

opts.batchSize        = 32;
opts.gpus             = [1]; %%% this code can only support one GPU!
opts.numSubBatches    = 2;
opts.bnormLearningRate= 0;
           
%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;
opts.imdbDir          = 'data/model_MC_Res_Bnorm_Adam/imdb_LDR.mat';
opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%------------;-------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
net  = feval(['DnCNN_init_',opts.modelName]);
%%%  load data
opts.expDir      = fullfile('data', opts.modelName);
%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------
[net, info] = DnCNN_train(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'bnormLearningRate',opts.bnormLearningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'numberImdb',opts.numberImdb, ...
    'backPropDepth',opts.backPropDepth, ...
    'imdbDir',opts.imdbDir, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;

trainingScenes = dir('data\TrainMat\data\');
trainingScenes = trainingScenes(3:end);
diary(fullfile(opts.expDir,'trainIfor.txt'));
diary on;
disp (datestr(now));
disp 'training scenes: ';
disp ([trainingScenes.name]);
disp 'learning rates: ';
disp (opts.learningRate);
disp 'net architecture: ';
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;
disp 'loss function: ';
lossF = importdata('vl_nnloss.m');
disp(lossF);
diary off;


