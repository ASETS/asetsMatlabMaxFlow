function [] = smoothManualSegmentations2D()

%   Martin Rajchl, Imperial College London, 2015
%   Example application: Regularization of manual segmentations 
%   
%   Flawed manual segmentations are regularized [1] with constant and 
%   contrast sensitive regularization terms [2].
%
%   References: 
%
%   [1] Yuan, J.; Bae, E.; Tai, X.-C.; Boycov, Y.
%       A Continuous Max-Flow Approach to Potts Model
%       ECCV, 2010
%
%   [2] Rajchl M., et al. (2014). Interactive Hierarchical Max-Flow 
%       Segmentation of Scar Tissue from Late-Enhancement Cardiac MR Images. 
%       IEEE Transactions on Medical Imaging 33(1), 159–172.


close all;
clear all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
visualizationFLAG = 1;

% load image and user scribbles
load(['..', filesep, 'data', filesep, 'brain_1125.mat'], 'img', 'man_s40_flawed');

man = man_s40_flawed(:,:,40);
img = img(:,:,40);


labelIds = sort(unique(man),'ascend');

% parameters
numberOfLabels = length(labelIds);
[r, c] = size(img);

% alloc a cost function Ct for each label i, int lId
Ct = zeros(r,c, numberOfLabels);

% cast to float for smoothing 
man = single(man);

% create a Gaussian kernel
hs=fspecial('gaussian',[5,5],1.5);

% compute data term from Gaussian smoothed manual segmentations
for i=1:numberOfLabels
    inv_bin_label = (1- single(man == labelIds(i)));
   Ct(:,:,i) = imfilter(inv_bin_label,hs,'replicate'); 
end

% pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
pars = [r; c; numberOfLabels; 300; 1e-11; 0.25; 0.11];


% Regularize with constant regularization term alpha 
alpha1 = 0.2.*ones(r,c, numberOfLabels);

% call 3D max-flow optimizer
[u, erriter, i, timet] = asetsPotts2D(single(Ct), single(alpha1), single(pars));

% maj vote to discretize continuous labels
[uu,I] = max(u, [], 3);



% Regularize with contrast sensitive regularization term alpha(x)
img = (img - min(img(:)))/(max(img(:)) - min(img(:)));
% compute gradient magnitude from image
[gx, gy] = gradient(img);
gm = sqrt(gx.^2 + gy.^2);

% build contrast sensitive regularization as in [2]
for i=1:numberOfLabels
    alpha2(:,:,i) = 0.0 + 0.2.* exp(-10*gm);
end

% call 3D max-flow optimizer
[u2, erriter2, i2, timet2] = asetsPotts2D(single(Ct), single(alpha2), single(pars));

% maj vote to discretize continuous labels
[uu2,I2] = max(u2, [], 3);


% visualize
if(visualizationFLAG)
   
    figure();
    subplot(2,3,1); imshow(img,[]); title('img');
    subplot(2,3,2); imshow(alpha1(:,:,1),[]); title ('alpha const.');
    subplot(2,3,3); imshow(alpha2(:,:,1),[]); title ('alpha constrast sens.');
    subplot(2,3,4); imshow(man,[]); title('before');
    subplot(2,3,5); imshow(labelIds(I),[]); title('constant reg.');
    subplot(2,3,6); imshow(labelIds(I2),[]); title('contrast sensitive reg.');
        
end


end
