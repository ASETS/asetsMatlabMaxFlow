function [] = interactiveGraphCutSegmentation3D()

%   Martin Rajchl, Imperial College London, 2015
%   Example application: Interactive graph cut segmentation 
%   
%   A log-likelihood cost of intensities sampled via user scribbles is used
%   to segment the background, cerebro-spinal fluid (csf), white matter
%   (wm) and gray matter (gm) from a T1-weighted brain MR image.
%
%   References: 
%   [1] Rajchl M., et al. (2014). Interactive Hierarchical Max-Flow 
%       Segmentation of Scar Tissue from Late-Enhancement Cardiac MR Images. 
%       IEEE Transactions on Medical Imaging 33(1), 159–172.
%
%   [2] Baxter, JSH. et al (2015). Optimization-Based Interactive 
%       Segmentation Interface for Multi-Region Problems. 
%       SPIE Medical Imaging. 

close all;
clear all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
visualizationFLAG = 1;

% load image and user scribbles
load(['..', filesep, 'data', filesep, 'brain_1125.mat'], 'img', 'scribbles');

labelIds = unique(scribbles(scribbles ~= 0));

% parameters
alpha1 = 0.025;
numberOfLabels = length(labelIds);
[r, c, s] = size(img);

% alloc a cost function Ct for each label i, int lId
Ct = zeros(r,c,s, numberOfLabels);
alpha = alpha1.*ones(r,c,s, numberOfLabels);

% compute the likelihood from the probabilities \in [0,1] as data term
epsilon = 1e-10;

for i=1:numberOfLabels
   Ct(:,:,:,i) = computeLogLikelihoodCost(img, scribbles == i, epsilon);
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
    nVis = numberOfLabels+2;
    for i=1:nVis
        switch i
            case 1
                subplot(3,nVis,i); imshow(squeeze(img(vis_r,:,:)),[]); title('img');
                subplot(3,nVis,i+nVis); imshow(squeeze(img(:,vis_c,:)),[]);
                subplot(3,nVis,i+2*nVis); imshow(squeeze(img(:,:,vis_s)),[]);
            case 2
                subplot(3,nVis,i); imshow(squeeze(I(vis_r,:,:)),[1 numberOfLabels]); title('seg');
                subplot(3,nVis,i+nVis); imshow(squeeze(I(:,vis_c,:)),[1 numberOfLabels]);
                subplot(3,nVis,i+2*nVis); imshow(squeeze(I(:,:,vis_s)),[1 numberOfLabels]);
            otherwise
                subplot(3,nVis,i); imshow(squeeze(u(vis_r,:,:,i-2)),[0 1]); title(['u_',num2str(i-2)]);
                subplot(3,nVis,i+nVis); imshow(squeeze(u(:,vis_c,:,i-2)),[0 1]);
                subplot(3,nVis,i+2*nVis); imshow(squeeze(u(:,:,vis_s,i-2)),[0 1]);
        end
    end
        
end


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




