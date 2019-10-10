function net = assembleResNet101()
% assembleResNet101   Assemble ResNet-101 network
%
% net = assembleResNet101 creates a ResNet-101 network with weights trained
% on ImageNet. You can load the same ResNet-101 network by installing the
% Deep Learning Toolbox Model for ResNet-101 Network support package from
% the Add-On Explorer and then using the resnet101 function.

%   Copyright 2019 The MathWorks, Inc.

% Download the network parameters. If these have already been downloaded,
% this step will be skipped.
%
% The files will be downloaded to a file "resnet101Params.mat", in a
% directory "ResNet101" located in the system's temporary directory.
dataDir = fullfile(tempdir, "ResNet101");
paramFile = fullfile(dataDir, "resnet101Params.mat");
downloadUrl = "http://www.mathworks.com/supportfiles/nnet/data/networks/resnet101Params.mat";

if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

if ~exist(paramFile, "file")
    disp("Downloading pretrained parameters file (160 MB).")
    disp("This may take several minutes...");
    websave(paramFile, downloadUrl);
    disp("Download finished.");
else
    disp("Skipping download, parameter file already exists.");
end

% Load the network parameters from the file resNet101Params.mat.
s = load(paramFile);
params = s.params;
paramNames = fields(params);

% Create a layer graph with the network architecture of ResNet-101.
lgraph = resnet101Layers;

% Create a cell array containing the layer names.
layerNames = {lgraph.Layers(:).Name}';

% Specify the average image of the image input layer. To specify properties
% of layers in the layer graph, you must create a copy of the layer,
% specify the property of the new layer, and then replace the old layer
% with the new one.
layer = lgraph.Layers(1);
layer.AverageImage = params.AverageImage;
lgraph = replaceLayer(lgraph,layerNames{1},layer);

% Specify the weights of the first convolutional layer.
layer = lgraph.Layers(2);
layer.Weights = params.conv1.weights;
layer.Bias = zeros(1,1,size(params.conv1.weights,4));
lgraph = replaceLayer(lgraph,layerNames{2},layer );

% Specify the weights of the final fully connected layer at the end of the
% network. The last fully connected layer outputs class scores.
layer = lgraph.Layers(end-2);
layer.Weights = params.fc1000.weights;
layer.Bias = params.fc1000.biases;
lgraph = replaceLayer(lgraph,layerNames{end-2},layer);

% Specify the classes of the classification layer.
layer = classificationLayer("Classes",params.categoryNames,"Name",layerNames{end});
lgraph = replaceLayer(lgraph,layerNames{end},layer);

% Specify the parameters of the convolutional and batch normalization
% layers.
learnableLayerNames = intersect(layerNames,paramNames);
for i = 1:numel(learnableLayerNames)
    name = learnableLayerNames{i};
    idx = strcmp(layerNames,name);
    layer = lgraph.Layers(idx);
    
    if isa(layer,"nnet.cnn.layer.Convolution2DLayer")
        % Convolutional layers have learned Weights parameters. The Bias
        % parameters are all zero.
        layerParams = params.(name);
        layer.Weights = layerParams.weights;
        layer.Bias = zeros(1,1,size(layerParams.weights,4));
        
    elseif isa(layer,"nnet.cnn.layer.BatchNormalizationLayer")
        % Batch normalization layers have TrainedMean and TrainedVariance
        % variables and learned Offset and Scale parameters.
        trainedVars = params.(name);
        layer.TrainedMean = reshape(trainedVars.trainedMean,1,1,[]);
        layer.TrainedVariance = reshape(trainedVars.trainedVariance,1,1,[]);
        
        % Learned Offset and Scale parameters are stored as a vector e.g
        % with size 256x1. Reshape to a three-dimensional array e.g. with
        % size 1x1x256.
        learnedParams = params.(replace(name,"bn","scale"));
        layer.Offset = reshape(learnedParams.offset,1,1,[]);
        layer.Scale = reshape(learnedParams.scale,1,1,[]);
    end
    
    lgraph = replaceLayer(lgraph,name,layer);
end

% Assemble the network.
net = assembleNetwork(lgraph);

end