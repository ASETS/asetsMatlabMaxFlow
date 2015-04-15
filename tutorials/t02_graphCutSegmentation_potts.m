%% Tutorial 02: Multi-region graph cut color segmentation with the Potts model
%  Martin Rajchl, Imperial College London, 2015
%
%   [1] Yuan, J.; Bae, E.; Tai, X.-C.; Boykov, Y.
%       A Continuous Max-Flow Approach to Potts Model
%       ECCV, 2010
%
%   [2] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Continuous Max-Flow Approach to General
%       Hierarchical Multi-Labelling Problems
%       arXiv preprint arXiv:1404.0336

  
clear all; close all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% 1. Load a color image and cast to single:
load('../data/natural_imgs.mat','berkeley_color_124084');
img = berkeley_color_124084;
imgs = single(img);

% 2. Create N = 4 cost functions to model each of the regions:
numberOfLabels = 4;
[sx, sy, rgb] = size(img);

% The costs are defined as L1 distances of the rgb image to N color models:
model(1,:) = [225,0,0]; % red
model(2,:) = [0,150,50]; % green
model(3,:) = [255,225,0]; % yellow
model(4,:) = [0,0,0]; % black

% compute the L1 cost term for each of the N = 4 regions:
cost = zeros(sx, sy, numberOfLabels,class(imgs));
for i=1:numberOfLabels
    cost(:,:,i) = (abs(imgs(:,:,1) - model(i,1)) + abs(imgs(:,:,2) - model(i,2)) + abs(imgs(:,:,3) - model(i,3)));
end

% visualize them
figure(); 
subplot(2,3,1); imshow(img,[]); title('image');
subplot(2,3,2); imshow(cost(:,:,1),[]); title('cost_{red}');
subplot(2,3,3); imshow(cost(:,:,2),[]); title('cost_{green}');
subplot(2,3,5); imshow(cost(:,:,3),[]); title('cost_{yellow}');
subplot(2,3,6); imshow(cost(:,:,4),[]); title('cost_{black}');

% 4. Construct the multi-label graph:
% allocate the sink links Ct(x)
Ct = zeros(sx,sy, numberOfLabels,class(imgs));

% allocate alpha(x), the regularization weight at each node x
alpha = zeros(sx,sy, numberOfLabels,class(imgs));

% 5. Assign capacities to the graph:
% Since this is a multi-label graph, where the source flows ps(x) are unconstrained,
% we define our sink capacities Ct(x,l) for each label l:

for i=1:numberOfLabels
    Ct(:,:,i) = cost(:,:,i);
end

% Assign a regularization weight (equivalent to pairwise terms) for each
% node x. Here we employ a constant regularization weight alpha(x,l). 
% Note, that the original Potts model does not have alpha indexed by label.
% This implementation is more flexible and represents a special case of 
% the hierarchical max flow, the horizontal model. However, in this case 
% we employ the orignial Potts model to multi-region segmentation 
% with constant regularization. Further, the regularization weight needs to
% be in a similar order of magnitude as Ct(x,l).
alpha = 0.025*max(cost(:)).*ones(sx,sy,numberOfLabels,class(imgs));
        
% 6. Set up the parameters for the max flow optimizer:
% [1] graph dimension 1
% [2] graph dimension 2
% [3] number of labels
% [4] number of maximum iterations for the optimizer (default 300)
% [5] an error bound at which we consider the solver converged (default
%     1e-5)
% [6] c parameter of the multiplier (default 0.2)
% [7] step size for the gradient descent step when calulating the spatial
%     flows p(x) (default 0.16)
pars = [sx; sy; numberOfLabels; 200; 1e-5; 0.2; 0.16];
    
% 7. Call the Potts model max flow optimizer Ct(x,l), alpha(x) and pars to obtain
% the continuous labelling function u(x,l), the convergence over iterations
% (conv), the number of iterations (numIt) and the run time (time);
[u, conv, numIt, time] = asetsPotts2D(Ct, alpha, pars);

% 8. To discretize the continuous labelling function u(x,l) we employ a
% majorty vote over l to obtain a final segmentation:
[tmp, idx] = max(u, [], 3);

% 9. Visualize the orignial image and the segmentation with the original
% color models:
seg = zeros(sx, sy, rgb, 'uint8');
for x=1:sx
    for y=1:sy
        seg(x,y,:) = model(idx(x,y),:);
    end
end
figure(); 
subplot(1,2,1); imshow(img,[]); title('Original image');
subplot(1,2,2); imshow(seg,[]); title('Segmentation u_{discrete}');

