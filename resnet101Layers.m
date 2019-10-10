function lgraph = resnet101Layers()
% resnet101Layers  ResNet-101 layer graph
%
% lgraph = resnet101Layers creates a layer graph with the network
% architecture of ResNet-101. The layer graph contains no weights.

%   Copyright 2019 The MathWorks, Inc.

% Create the first five layers of the network. The network has an image
% input size of 224-by-224-by-3.
initialSection = [
    imageInputLayer([224 224 3],"Name","data")
    convolution2dLayer(7,64,"Stride",2,"Padding",3,"BiasLearnRateFactor",0,"Name","conv1")
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer(3,"Stride",2,"Padding",[0 1 0 1],"Name","pool1")];
lgraph = layerGraph(initialSection);

% Use the function addAndConnectResNetSection to create a ResNet section
% and connect it to the end of the layer graph.
lgraph = addAndConnectResNetSection(lgraph, ...
    lgraph.Layers(end).Name, ...    % Layer to connect to
    "res2", ...                     % Name of ResNet section
    64, ...                         % Number of filters in the first 1-by-1 convolutions
    64, ...                         % Number of filters in the 3-by-3 convolutions
    256, ...                        % Number of filters in the last 1-by-1 convolutions
    1, ...                          % Stride of the first convolution of the section
    {'a','b','c'});                 % Names of the residual units

% Add and connect three more ResNet sections to the network.
lgraph = addAndConnectResNetSection(lgraph, ...
    lgraph.Layers(end).Name, ...
    "res3", ...
    128, ...
    128, ...
    512, ...
    2, ...
    {'a','b1','b2','b3'});

lgraph = addAndConnectResNetSection(lgraph, ...
    lgraph.Layers(end).Name, ...
    "res4", ...
    256, ...
    256, ...
    1024, ...
    2, ...
    {'a','b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12',...
    'b13','b14','b15','b16','b17','b18','b19','b20','b21','b22'});

lgraph = addAndConnectResNetSection(lgraph, ...
    lgraph.Layers(end).Name, ...
    "res5", ...
    512, ...
    512, ...
    2048, ...
    2, ...
    {'a','b','c'});

% Create, add, and connect the final ResNet section of the network.
finalSection = [
    averagePooling2dLayer(7,"Stride",7,"Name","pool5")
    fullyConnectedLayer(1000,"Name","fc1000")
    softmaxLayer("Name","prob")
    classificationLayer("Name","ClassificationLayer_predictions")];

lastLayerName = lgraph.Layers(end).Name;
lgraph = addLayers(lgraph,finalSection);
lgraph = connectLayers(lgraph,lastLayerName,"pool5");

end


function lgraph = addAndConnectResNetSection( ...
    lgraph, ...
    layerToConnectFrom, ...
    sectionName, ...
    numF1x1first, ...
    numF3x3, ...
    numF1x1last, ...
    firstStride, ...
    residualUnits)
% addAndConnectResNetSection   Creates a ResNet section and connects it to
% the specified layer of a layer graph.
%
% This function connects a new ResNet section to the layer
% layerToConnectFrom in the layer graph lgraph.
%
% sectionName is the name of the new network section.
%
% numF1x1first is the number of filters in the first 1-by-1 convolution of
% the residual units of the section.
%
% numF3x3 is the number of filters in the 3-by-3 convolution of the
% residual units of the section.
%
% numF1x1last is the number of filters in the last 1-by-1 convolution of
% the residual units of the section.
%
% firstStride is the stride of the first convolution of the section.
%
% residualUnits contains the names of the residual units of the section.
% This argument determines the number of residual units of the
% section.
stride = firstStride;
for i = 1:length(residualUnits)   
    layerRoot = sectionName+residualUnits{i};
    bnRoot = "bn"+extractAfter(sectionName,"res")+residualUnits{i};
    
    % Create the block of layers for the residual unit and add it to the
    % layer graph.
    resnetBlock = [
        convolution2dLayer(1,numF1x1first,"Stride",stride,"Name",layerRoot+"_branch2a","BiasLearnRateFactor",0)
        batchNormalizationLayer("Name",bnRoot+"_branch2a")
        reluLayer("Name",layerRoot+"_branch2a_relu")
        
        convolution2dLayer(3,numF3x3,"Padding",1,"Name",layerRoot+"_branch2b","BiasLearnRateFactor",0);
        batchNormalizationLayer("Name",bnRoot+"_branch2b")
        reluLayer("Name",layerRoot+"_branch2b_relu")
        
        convolution2dLayer(1,numF1x1last,"Name",layerRoot+"_branch2c","BiasLearnRateFactor",0)
        batchNormalizationLayer("Name",bnRoot+"_branch2c")
        
        additionLayer(2,"Name",layerRoot)
        reluLayer("Name",layerRoot+"_relu")];
    
    lgraph = addLayers(lgraph,resnetBlock);
    
    % For the first residual unit of the section, add a residual connection
    % with projection layers and connect to the previous section.
    % Projection layers contain a 1-by-1 convolutional layer that projects
    % the input into a space with a larger number of channels. This
    % projection is necessary because the activations in different sections
    % of the network have a different number of channels. For the other
    % residual units, connect to the previous residual unit.
    if i == 1
        projectionLayers = [
            convolution2dLayer(1,numF1x1last,"Stride",stride,"Name",layerRoot+"_branch1","BiasLearnRateFactor",0)
            batchNormalizationLayer("Name",bnRoot+"_branch1")];
        lgraph = addLayers(lgraph,projectionLayers);
        
        lgraph = connectLayers(lgraph,layerToConnectFrom,layerRoot+"_branch2a");
        lgraph = connectLayers(lgraph,layerToConnectFrom,layerRoot+"_branch1");
        lgraph = connectLayers(lgraph,bnRoot+"_branch1",layerRoot+"/in2");
    else
        lgraph = connectLayers(lgraph,sectionName+residualUnits{i-1}+"_relu",sectionName+residualUnits{i}+"_branch2a");
        lgraph = connectLayers(lgraph,sectionName+residualUnits{i-1}+"_relu",sectionName+residualUnits{i}+"/in2");
    end
    
    stride = 1;
end
end