function [] = L1IntensitySegmentation2D()

%   Martin Rajchl, Imperial College London, 2015
%   Example application: Intensity segmentation with an L1 data term
%   
%   References: 
%   [1] Yuan, J.; Bae, E.; Tai, X.-C.; Boykov, Y.
%       A Continuous Max-Flow Approach to Potts Model
%       ECCV, 2010
%

close all;
clear all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
visualizationFLAG = 1;

% load image
load(['..', filesep, 'data', filesep, 'brain_1125.mat'], 'img', 'man_s40_flawed');

img = img(:,:,40);

% parameters
alpha1 = 0.025;
numberOfLabels = 4;
[r, c] = size(img);

% alloc a cost function Ct for each label i, int lId
Ct = zeros(r,c, numberOfLabels);
alpha = alpha1.*ones(r,c, numberOfLabels);

% normalize image
img = (img - min(img(:)))/ (max(img(:)) - min(img(:)) );

% assign models of mean intensity for each of the N regions
imgModels = [0, 0.25, 0.5, 0.9];

% compute intensity L1 data term
for i=1:numberOfLabels
   Ct(:,:,i) = abs(img - imgModels(i));
end

% pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
pars = [r; c; numberOfLabels; 200; 1e-11; 0.25; 0.11];

% call 2D max-flow optimizer
[u, erriter, i, timet] = asetsPotts2D(Ct, alpha, pars);

% maj vote to discretize continuous labels
[uu,I] = max(u, [], 3);

% visualize
if(visualizationFLAG)

    figure();
    subplot(1,2,1); imshow(img,[]); title('img');
    subplot(1,2,2); imshow(I,[]); title('L1 intensity seg');
        
end


end





