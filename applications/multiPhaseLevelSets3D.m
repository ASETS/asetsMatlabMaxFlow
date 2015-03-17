function [] = multiPhaseLevelSets3D()

%   Martin Rajchl, Imperial College London, 2015
%   Example application: Fast max flow propagated levelsets
%
%
%   References:
%   [1] Rajchl M., et al. (2015). Variational Time-Implicit Multiphase
%   Level-Sets: A Fast Convex Optimization-Based Solution.
%   In: Energy Minimization Methods in Computer Vision and
%   Pattern Recognition. Springer Berlin Heidelberg, pp.278–291.

close all;
clear all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
useCUDAFLAG = 1;
visualizationFLAG = 1;


initType = 'spheres'; % alternatively use 'cubes', 'custom'

% load image and inital regions
load(['..', filesep, 'data', filesep, 'brain_1125.mat'], 'img', 'lset_init_cubes','lset_init_custom','lset_init_spheres');

switch initType
    case 'spheres'
        initRegions = lset_init_spheres;
    case 'cubes'
        initRegions = lset_init_cubes;
    case 'custom'
        initRegions = lset_init_custom;
    otherwise
        return;
end

labelIds = unique(initRegions);

% parameters
w1 = 0.6; % weight for the intensity term
w2 = 0.4; % weight for the speed term
alpha1 = 0.1; % regularization weight
tau = 25; % speed parameter
epsilon = 1e-10;

maxLevelSetIterations = 10; % maximum number of levelset iterations
numberOfLabels = length(labelIds); 
[r, c, s] = size(img);

% alloc a data term Ct and alpha
Ct = zeros(r,c,s, numberOfLabels);
alpha = alpha1.*ones(r,c,s, numberOfLabels);

% pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
pars = [r; c; s; numberOfLabels; 200; 1e-5; 0.25; 0.11];

% normalize image
imax = max(img(:));
imin = min(img(:));
img = (img - imin)/(imax-imin);

currRegions = single(initRegions);
% levelset iterations
for it=1:maxLevelSetIterations
    
    for i=1:numberOfLabels
        
        region = currRegions == labelIds(i);
        
        % Compute intensity data term
        d_int = computeLogLikelihoodCost(img, region, epsilon);
        
        % Compute speed term
        d_dist_in = -bwdist(1- region,'euclidean').*region;
        d_dist_out = bwdist(region,'euclidean').*(1 - region);
        
        d_speed = d_dist_in + d_dist_out;
        
        % Combine and re-weight
        Ct(:,:,:,i) = w1.*d_int + w2.*(d_speed./tau);
        
    end
    
    % normalize Ct
    Ct = Ct - min(min(min(min(Ct))));
    
    % call 3D max-flow optimizer with CUDA if possible
    if(gpuDeviceCount && useCUDAFLAG)
        [u, erriter, i, timet] = asetsPotts3D(gpuArray(Ct), gpuArray(alpha), pars);
    else
        [u, erriter, i, timet] = asetsPotts3D(Ct, alpha, pars);
    end
    
    % maj vote to discretize continuous labels
    [uu,I] = max(u, [], 4);
    
    % assign currRegions for subsequent iterations
    currRegions = labelIds(I);    
    
    % visualize
    if(visualizationFLAG)
        vis_s = 40;
        close all;
        figure();
        subplot(1,3,1); imshow(img(:,:,vis_s),[]);
        subplot(1,3,2); imshow(initRegions(:,:,vis_s),[0 numberOfLabels]);
        subplot(1,3,3); imshow(currRegions(:,:,vis_s),[0 numberOfLabels]);
        colormap('jet');
        drawnow();
    end
    
end

% visualize
if(visualizationFLAG)
    
    % compute mid slices in each direction
    vis_r = idivide(r,uint8(2));
    vis_c = idivide(c,uint8(2));
    vis_s = idivide(s,uint8(2));
    
    figure();
    nVis = numberOfLabels+2;
    for i=1:nVis
        switch i
            case 1
                subplot(3,nVis,i); imshow(squeeze(img(vis_r,:,:)),[]); title('img');
                subplot(3,nVis,i+nVis); imshow(squeeze(img(:,vis_c,:)),[]);
                subplot(3,nVis,i+2*nVis); imshow(squeeze(img(:,:,vis_s)),[]);
            case 2
                subplot(3,nVis,i); imshow(squeeze(currRegions(vis_r,:,:)),[1 numberOfLabels]); title('seg');
                subplot(3,nVis,i+nVis); imshow(squeeze(currRegions(:,vis_c,:)),[1 numberOfLabels]);
                subplot(3,nVis,i+2*nVis); imshow(squeeze(currRegions(:,:,vis_s)),[1 numberOfLabels]);
            otherwise
                subplot(3,nVis,i); imshow(squeeze(u(vis_r,:,:,i-2)),[0 1]); title(['u_',num2str(i-2)]);
                subplot(3,nVis,i+nVis); imshow(squeeze(u(:,vis_c,:,i-2)),[0 1]);
                subplot(3,nVis,i+2*nVis); imshow(squeeze(u(:,:,vis_s,i-2)),[0 1]);
        end
    end
    
end


end

function [cost] = computeChanVeseCost(img, lbl)

cost = (img - mean(mean(mean(img(lbl))))).^2;

end

function [cost] = computeLogLikelihoodCost(img, lbl, epsilon)

img = single(img);

nBins = 256;

minI = min(img(:));
maxI = max(img(:));

% normalize image to 8 bit
img_n = ((img - minI)/ (maxI - minI)).*255.0;

% compute histogram
[binCounts] = histc(img_n(lbl == 1),linspace(0,255,nBins));

% normalize to compute the probabilities
binCounts = binCounts./sum(binCounts(:));

% compute LL
P = binCounts( uint16(img_n/ (256/nBins)) + 1);
cost = -log10(P  + epsilon);

end






