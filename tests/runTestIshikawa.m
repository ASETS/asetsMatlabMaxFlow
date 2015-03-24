function [] = runTestIshikawa()

%   Martin Rajchl, Imperial College London, 2015
%   Testing scripts for Ishikawa model regularization

close all;
clear all;

addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
run2DIshikawaTestFLAG = 1;
run3DIshikawaTestFLAG = 1;

visualizationFLAG = 1;

% parameters
numberOfLabels = 6;
r = 64; % number of rows
c = 64; % number of columns
s = 10; % number of slices

maxIter = 500; % maximum number of max flow iterations
convErrBound2D = 1e-11; % bound at which the max flow is considered converged 
convErrBound3D = 1e-11; % bound at which the max flow is considered converged 

if (run2DIshikawaTestFLAG)
    
    % alloc a cost function Ct for each label i, int lId
    Ct = zeros(r,c, numberOfLabels);
    alpha = zeros(r,c, numberOfLabels-1);
    
    % for each label assign a random data cost
    for i=1:numberOfLabels
        rng shuffle;
        Ct(:,:,i) = rand(r,c);
        h = fspecial('gaussian', [1 17], 1);
        Ct(:,:,i) = imfilter(Ct(:,:,i),h);
    end
    
    % for each label assign a constant regularization weight
    for i=1:(numberOfLabels-1)
        alpha(:,:,i) = (0.8/i).*ones(r,c);
    end
    
    % call max-flow optimizer
    % pars = [rows; columns; numberOfLabels; maxIter; convRate; cc; stepSize];
    pars = [r; c; numberOfLabels; maxIter; convErrBound2D; 0.75; 0.16];
    
    % run both 2D matlab and mex implementations
    [u, erriter, i, timet] = asetsIshikawa2D_mex(single(Ct), single(alpha), single(pars));
    [u2, erriter2, i2, timet2] = asetsIshikawa2D(single(Ct), single(alpha), single(pars));
    
    
    % threshold discretize continuous labels
    ut = zeros(r,c);
    for k=1:numberOfLabels-1
        ut = ut + (u(:,:,k) > 0.5);
    end
    
    u2t = zeros(r,c);
    for k=1:numberOfLabels-1
        u2t = u2t + (u2(:,:,k) > 0.5);
    end
    
    
    % visualize
    if (visualizationFLAG)
        
        figure();
        for i=1:(numberOfLabels-1)
            subplot(3,numberOfLabels,i); imshow(Ct(:,:,i),[]); title(['Ct_',num2str(i)]);
            subplot(3,numberOfLabels,i+numberOfLabels); imshow(u(:,:,i),[0 1]); title(['mex/C: u_',num2str(i)]);
            subplot(3,numberOfLabels,i+2*numberOfLabels); imshow(u2(:,:,i),[0 1]); title(['Matlab/CUDA: u_',num2str(i)]);
        end
        
        % view resulting labeling functions from each implementation
        figure();
        subplot(1,2,1); imshow(ut,[1 numberOfLabels]); title('mex/C: u_{discrete}');
        subplot(1,2,2); imshow(u2t,[1 numberOfLabels]); title('Matlab/CUDA: u_{discrete}');
        
        implErr = 0;
        for i = 1:(numberOfLabels-1)
            implErr = implErr + sum(sum(abs((ut == i) - (u2t == i))));
        end
        disp(['Labeling error between implementations = ', num2str(implErr)]);
        
        % convergence plots
        figure();
        subplot(1,2,1); loglog(erriter); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence mex/C');
        subplot(1,2,2); loglog(erriter2); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence Matlab/CUDA');
        
        
    end
    colormap('jet');
    
end



if (run3DIshikawaTestFLAG)
    
    % alloc a cost function Ct for each label i, int lId
    Ct = zeros(r,c,s, numberOfLabels);
    alpha = zeros(r,c,s, numberOfLabels);
    
    % for each label assign a random data cost
    for i=1:numberOfLabels
        rng shuffle;
        Ct(:,:,:,i) = rand(r,c,s);
        h = fspecial('gaussian', [1 17], 2);
        for j=1:s
            Ct(:,:,j,i) = imfilter(Ct(:,:,j,i),h);
        end
    end
    
    % for each label assign a constant regularization weight
    for i=1:numberOfLabels
        alpha(:,:,:,i) = (0.3/i).*ones(r,c,s);
    end
    
    
    % call 3D max-flow optimizer
    
    % pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
    pars = [r; c; s; numberOfLabels; maxIter; convErrBound3D; 0.75; 0.1];
    
    % run both 3D matlab and mex implementations
    [u, erriter, i, timet] = asetsIshikawa3D_mex(single(Ct), single(alpha), single(pars));
    [u2, erriter2, i2, timet2] = asetsIshikawa3D(single(Ct), single(alpha), single(pars));
    
    
    % threshold discretize continuous labels
    ut = zeros(r,c,s);
    for k=1:numberOfLabels-1
        ut = ut + (u(:,:,:,k) > 0.5);
    end
    
    u2t = zeros(r,c,s);
    for k=1:numberOfLabels-1
        u2t = u2t + (u2(:,:,:,k) > 0.5);
    end
    
    % visualize
    if(visualizationFLAG)
        
        % compute mid slices in each direction
        vis_r = idivide(r,uint8(2));
        vis_c = idivide(c,uint8(2));
        vis_s = idivide(s,uint8(2));
        
        figure();
        for i=1:(numberOfLabels-1)
            subplot(4,numberOfLabels,i); imshow(Ct(:,:,vis_s,i),[]); title(['Ct_',num2str(i)]);
            subplot(4,numberOfLabels,i+numberOfLabels); imshow(squeeze(u(:,:,vis_s,i)),[0 1]); title(['mex/C: u_',num2str(i)]);
            subplot(4,numberOfLabels,i+2*numberOfLabels); imshow(squeeze(u2(:,:,vis_s,i)),[0 1]); title(['Matlab/CUDA: u_',num2str(i)]);
        end
        
        % view resulting labeling functions from each implementation
        figure();
        subplot(2,3,1); imshow(squeeze(ut(vis_r,:,:)),[1 numberOfLabels]); title('mex/C: u_{discrete}');
        subplot(2,3,2); imshow(squeeze(ut(:,vis_c,:)),[1 numberOfLabels]);
        subplot(2,3,3); imshow(squeeze(ut(:,:,vis_s)),[1 numberOfLabels]);
        subplot(2,3,4); imshow(squeeze(u2t(vis_r,:,:)),[1 numberOfLabels]); title('Matlab/CUDA: u_{discrete}');
        subplot(2,3,5); imshow(squeeze(u2t(:,vis_c,:)),[1 numberOfLabels]);
        subplot(2,3,6); imshow(squeeze(u2t(:,:,vis_s)),[1 numberOfLabels]);
        
        colormap('jet');
        implErr = 0;
        for i = 1:(numberOfLabels-1)
            implErr = implErr + sum(sum(sum(abs((ut == i) - (u2t == i)))));
        end
        disp(['Labeling error between implementations = ', num2str(implErr)]);
        
        % convergence plots
        figure();
        subplot(1,2,1); loglog(erriter); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence mex/C');
        subplot(1,2,2); loglog(erriter2); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence Matlab/CUDA');
        
    end
end


end




