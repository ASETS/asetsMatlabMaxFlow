%% Tutorial 05: Time-implicit level set segmentation
%  Martin Rajchl, Imperial College London, 2015
%
%   [1] Rajchl, M.; Baxter, JSH.; Bae, E.; Tai, X-C.; Fenster, A.; 
%       Peters, TM.; Yuan, J.;
%       Variational Time-Implicit Multiphase Level-Sets: A Fast Convex 
%       Optimization-Based Solution
%       EMMCVPR, 2015.
%
%   [2] Ukwatta, E.; Yuan, J.; Rajchl, M.; Qiu, W.; Tessier, D; Fenster, A.
%       3D Carotid Multi-Region MRI Segmentation by Globally Optimal 
%       Evolution of Coupled Surfaces
%       IEEE Transactions on Medical Imaging, 2013


clear all; close all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% 1. Load an image to segment
load('../data/natural_imgs.mat','cameraman');
img = cameraman;

% 2. Normalize the image intensity to [0,1]:
img = single(img);
img_n = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% 3. Initialize a region as initialization for the zero level set
region = zeros(size(img_n),'like', img_n);
region(64:196,64:196) = 1;

% visualize initial region
figure();
subplot(2,2,1); imshow(img_n,[]);
hold on; contour(region,'r'); hold off;
title('Initial region');

% 4. Construct an s-t graph:
[sx, sy] = size(img_n);

Cs = zeros(sx,sy);
Ct = zeros(sx,sy);

% allocate alpha(x), the regularization weight at each node x
alpha = zeros(sx,sy);

% 5. Set up parameters and start level set iterations:
maxLevelSetIterations = 20; % number of maximum time steps
tau = 50; % speed parameter
w1 = 0.6; % weight parameter for intensity data term
w2 = 0.4; % weight parameter for the speed data term
for t=1:maxLevelSetIterations
        
    % 6. Compute a speed data term based on the current region
    d_speed_inside = bwdist(region == 1,'Euclidean');
    d_speed_outside = bwdist(region == 0,'Euclidean');
    
    % 7. Compute a intensity data term based on the L1 distance to the
    % mean
    m_int_inside = mean(mean(img_n(region == 1)));
    m_int_outside =  mean(mean(img_n(region == 0)));
    
    d_int_inside = abs(img_n - m_int_inside);
    d_int_outside = abs(img_n - m_int_outside);
    
    % 8. Compute speed data term as in Tutorial 04:
    d_speed_inside = ((1-region).*d_speed_inside)./tau;
    d_speed_outside = (region.*d_speed_outside)./tau;
    
    % 7. Weight the contribution of both costs and assign them as source 
    % and sink capacities Cs, Ct in the graph
    Cs = w1.*d_int_outside + w2.*d_speed_outside;
    Ct = w1.*d_int_inside + w2.*d_speed_inside;
    
    % Assign a regularization weight (equivalent to pairwise terms) for each
    % node x. Here we employ a constant regularization weight alpha. The higher
    % alpha is, the more smoothness penalty is assigned.
    alpha = 1.5.*ones(sx,sy);
    
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
    region = u > 0.5;
   
    % visualize the costs
    subplot(2,2,2); imshow(Cs-Ct,[]); title('Cs-Ct');
    subplot(2,2,3); imshow(img,[]); title(['r(',num2str(t),')']); hold on; contour(region,'r'); hold off;
    subplot(2,2,4); loglog(conv); title(['convergence(',num2str(t),')']);
    drawnow();
        
end


