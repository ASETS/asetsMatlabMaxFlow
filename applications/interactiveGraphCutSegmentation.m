function [] = interactiveGraphCutSegmentation()

%   Martin Rajchl, Imperial College London, 2015
%   Example application: Interactive graph cut segmentation 

close all;
clear all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
visualizationFLAG = 1;

% load image
load(['..', filesep, 'data', filesep, 'brain_1125.mat']);

labelIds = unique(scribbles(scribbles ~= 0));

% parameters
alpha1 = 0.05;
numberOfLabels = length(labelIds);
[r, c, s] = size(img);

% alloc a cost function Ct for each label i, int lId
Ct = zeros(r,c,s, numberOfLabels);
alpha = alpha1.*ones(r,c,s, numberOfLabels);

% compute the likelihood from the probabilities \in [0,1] as data term
epsilon = 1e-10;

for i=1:numberOfLabels
    COMPUTE LL TERM HERE! 
   Ct(:,:,:,i) = 
    
end

% pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
pars = [r; c; s; numberOfLabels; 200; 1e-11; 0.25; 0.11];

% call 3D max-flow optimizer
[u, erriter, i, timet] = asetsPotts3D(Ct, alpha, pars);

% maj vote to discretize continuous labels
[uu,I] = max(u, [], 4);

% visualize
if(visualizationFLAG)
    
    % compute mid slices in each direction
    vis_r = idivide(r,uint8(2));
    vis_c = idivide(c,uint8(2));
    vis_s = idivide(s,uint8(2));
    
    figure();
    for i=1:(numberOfLabels)
        subplot(4,numberOfLabels,i); imshow(Ct(:,:,vis_s,i),[0 1]);
        subplot(4,numberOfLabels,i+numberOfLabels); imshow(squeeze(u(vis_r,:,:,i)),[0 1]);
        subplot(4,numberOfLabels,i+2*numberOfLabels); imshow(squeeze(u(:,vis_c,:,i)),[0 1]);
        subplot(4,numberOfLabels,i+3*numberOfLabels); imshow(squeeze(u(:,:,vis_s,i)),[0 1]);
    end
    
    % view resulting labeling functions from each implementation
    figure();
    subplot(1,3,1); imshow(squeeze(I(vis_r,:,:)),[1 numberOfLabels]);
    subplot(1,3,2); imshow(squeeze(I(:,vis_c,:)),[1 numberOfLabels]);
    subplot(1,3,3); imshow(squeeze(I(:,:,vis_s)),[1 numberOfLabels]);
    
    colormap('jet');
    
end


end




