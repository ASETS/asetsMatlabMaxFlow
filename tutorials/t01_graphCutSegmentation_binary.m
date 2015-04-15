%% Tutorial 01: Binary Graph cut segmentation
%  Martin Rajchl, Imperial College London, 2015
%
%   [1] Yuan, J.; Bae, E.; Tai, X,-C.;
%       A study on continuous max-flow and min-cut approaches
%       IEEE CVPR, 2010
%
%   [2] Rajchl, M.; Yuan, J.; Peters, TM.
%       Real-time segmentation in 4D ultrasound with 
%       continuous max-flow
%       SPIE Medical Imaging 2012, 83141F-83141F-8

  
clear all; close all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% 1. Load an 8-bit greyscale image:
load('../data/natural_imgs.mat','cameraman');
img = cameraman;

% 2. Normalize the image intensity to [0,1]:
img = single(img);
img_n = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% 3. Create two cost functions to model foreground (fg) and background (bg):

% The costs are defined as L1 distances of the image to two intensity 
% values we consider descriptive of fg and bg, respectively.
val_fg = 0.75; val_bg = 0.25;

% compute the L1 cost terms
cost_fg = abs(img_n - val_fg);
cost_bg = abs(img_n - val_bg);

% visualize them
figure(); 
subplot(1,2,1); imshow(cost_fg,[]); title('cost_{fg}');
subplot(1,2,2); imshow(cost_bg,[]); title('cost_{bg}');

% 4. Construct an s-t graph:
[sx, sy] = size(cost_fg);

% allocate s and t links
Cs = zeros(sx,sy);
Ct = zeros(sx,sy);

% allocate alpha(x), the regularization weight at each node x
alpha = zeros(sx,sy);

% 5. Assign capacities to the graph:
% Assign the costs from 3. as capacities for the s-t links as data
% consistency terms
Cs = cost_bg;
Ct = cost_fg;

% Assign a regularization weight (equivalent to pairwise terms) for each
% node x. Here we employ a constant regularization weight alpha. The higher
% alpha is, the more smoothness penalty is assigned.
alpha = 0.25.*ones(sx,sy);
        
% 6. Set up the parameters for the max flow optimizer:
% [1] graph dimension 1
% [2] graph dimension 2
% [3] number of maximum iterations for the optimizer (default 200)
% [4] an error bound at which we consider the solver converged (default
%     1e-5)
% [5] c parameter of the multiplier (default 0.2)
% [6] step size for the gradient descent step when calulating the spatial
%     flows p(x) (default 0.16)
pars = [sx; sy; 200; 1e-5; 0.2; 0.16];
    
% 7. Call the binary max flow optimizer with Cs, Ct, alpha and pars to obtain
% the continuous labelling function u, the convergence over iterations
% (conv), the number of iterations (numIt) and the run time (time);
[u, conv, numIt, time] = asetsBinaryMF2D(Cs, Ct, alpha, pars);

% 8. Threshold the continuous labelling function u to obtain a discrete
% segmentation result
ut = u > 0.5;

% 9. Visualize the orignial image and the segmentation 
figure(); 
subplot(1,3,1); imshow(img,[]); title('Original image');
subplot(1,3,2); imshow(u,[]); title('Segmentation: u_{continuous}');
subplot(1,3,3); imshow(ut,[]); title('Segmentation u_{discrete}');

