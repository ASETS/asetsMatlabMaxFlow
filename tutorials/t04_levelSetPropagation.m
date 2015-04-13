%% Tutorial 04: Time-implicit level set propagation with max flow
%  Martin Rajchl, Imperial College London, 2015

clear all; close all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% 1. Create domain Omega
omega = zeros(256,256);

% 2. Initialize a region as initialization for the zero level set
region = zeros(size(omega),'like', omega);
region(64:196,64:196) = 1;

% visualize initial region
figure();
subplot(2,2,1); imshow(omega,[]);
hold on; contour(region,'r'); hold off;
title('Initial region');

% 3. Construct an s-t graph:
[sx, sy] = size(omega);

Cs = zeros(sx,sy);
Ct = zeros(sx,sy);

% allocate alpha(x), the regularization weight at each node x
alpha = zeros(sx,sy);

% 4. Set up parameters and start level set iterations:
maxLevelSetIterations = 10; % number of maximum time steps
tau = 250; % speed parameter

for t=1:maxLevelSetIterations
        
    % 5. Compute a speed data term based on the current region
    d_speed_inside = bwdist(region == 1,'Euclidean');
    d_speed_outside = bwdist(region == 0,'Euclidean');
    
    % 6. Assign cost as source and sink capacities Cs, Ct in the graph
    Cs = (region.*d_speed_outside)./tau;
    Ct = ((1-region).*d_speed_inside)./tau;
    
    % Assign a regularization weight (equivalent to pairwise terms) for each
    % node x. Here we employ a constant regularization weight alpha. The higher
    % alpha is, the more smoothness penalty is assigned.
    alpha = 0.75.*ones(sx,sy);
    
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
    subplot(2,2,3); imshow(omega,[]); title(['r(',num2str(t),')']); hold on; contour(region,'r'); hold off;
    subplot(2,2,4); loglog(conv); title(['convergence(',num2str(t),')']);
    drawnow();
        
end


