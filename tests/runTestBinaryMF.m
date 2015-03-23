function [] = runTestBinaryMF()

%   Martin Rajchl, Imperial College London, 2015
%   Testing scripts for binary max flow regularization

close all;
clear all;

addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
run2DBinaryMFTestFLAG = 1;
run3DBinaryMFTestFLAG = 1;
run2DBinaryMF_starShapeTestFLAG = 0;

visualizationFLAG = 1;

% parameters
r = 64; % number of rows
c = 64; % number of columns
s = 32; % number of slices

maxIter = 300; % maximum number of iterations
convErrBound2D = 1e-5; % bound at which the max flow is considered converged 
convErrBound3D = 1e-5; % bound at which the max flow is considered converged 

if (run2DBinaryMFTestFLAG)
    
    % alloc the source and sink capacities Cs and Ct 
    Cs = zeros(r,c);
    Ct = zeros(r,c);
        
    % create random cost functions
    rng shuffle;
    Cs = rand(r,c);
    Ct = rand(r,c);
    
    h = fspecial('gaussian', [17 17], 1);
    Cs = imfilter(Cs,h);
    Ct = imfilter(Ct,h);
    
    % create constant pairwise costs
    alpha = 0.25.*ones(r,c);
        
    % call binary max-flow optimizer
    % pars = [rows; columns; maxIter; convRate; cc; stepSize];
    pars = [r; c; maxIter; convErrBound2D; 0.2; 0.16];
    
    % run both 2D matlab and mex implementations
    [u, erriter, numIt, timet] = asetsBinaryMF2D_mex(single(Cs), single(Ct), single(alpha), single(pars));
    [u2, erriter2, numIt, timet2] = asetsBinaryMF2D(Cs, Ct, alpha, pars);
    
    % threshold to discretize continuous labels
    I = u > 0.5;
    I2 = u2 > 0.5;
    
    % visualize
    if (visualizationFLAG)
        
        figure();
     
        subplot(2,2,1); imshow(Cs,[]); title('C_s')
        subplot(2,2,2); imshow(Ct,[]); title('C_t')
             
        % view resulting labeling functions from each implementation
        subplot(2,2,3); imshow(I,[0 1]); title('u_{mex}')
        subplot(2,2,4); imshow(I2,[0 1]); title('u_{Matlab}')
        
        disp(['Labelling error between implementations = ', num2str(sum(sum(abs(I-I2))))]);
        
        % convergence plots
        figure(); 
        subplot(1,2,1); loglog(erriter); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence mex/C');
        subplot(1,2,2); loglog(erriter2); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence Matlab/CUDA');
        
    end
    colormap('gray');
    
end



if (run3DBinaryMFTestFLAG)
    
    % alloc the source and sink capacities Cs and Ct 
    Cs = zeros(r,c,s);
    Ct = zeros(r,c,s);
        
     % create random cost functions
    rng shuffle;
    Cs = rand(r,c,s);
    Ct = rand(r,c,s);
    
    h = fspecial('gaussian', [17 17], 1);
    Cs = imfilter(Cs,h);
    Ct = imfilter(Ct,h);
    
    % create constant pairwise costs
    alpha = 0.1.*ones(r,c,s);
    
    % call 3D max-flow optimizer
    
    % pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
    pars = [r; c; s; maxIter; convErrBound3D; 0.25; 0.11];
    
    % run both 3D matlab and mex implementations
    [u, erriter, i, timet] = asetsBinaryMF3D_mex(single(Cs), single(Ct), single(alpha), single(pars));
    [u2, erriter2, i2, timet2] = asetsBinaryMF3D(Cs, Ct, alpha, pars);
    
    
    % threshold to discretize continuous labels
    I = u > 0.5;
    I2 = u2 > 0.5;
    
    % visualize
    if (visualizationFLAG)
        
        % compute mid slices in each direction
        vis_r = idivide(r,uint8(2));
        vis_c = idivide(c,uint8(2));
        vis_s = idivide(s,uint8(2));
        
        figure();
        subplot(3,4,1); imshow(squeeze(Cs(vis_r,:,:)),[]); title('C_s')
        subplot(3,4,2); imshow(squeeze(Ct(vis_r,:,:)),[]); title('C_t')
        subplot(3,4,3); imshow(squeeze(I(vis_r,:,:)),[]); title('u_{mex}')
        subplot(3,4,4); imshow(squeeze(I2(vis_r,:,:)),[]); title('u_{Matlab}')
             
        subplot(3,4,5); imshow(squeeze(Cs(:,vis_c,:)),[]);
        subplot(3,4,6); imshow(squeeze(Ct(:,vis_c,:)),[]); 
        subplot(3,4,7); imshow(squeeze(I(:,vis_c,:)),[]); 
        subplot(3,4,8); imshow(squeeze(I2(:,vis_c,:)),[]); 

        subplot(3,4,9); imshow(squeeze(Cs(:,:,vis_s)),[]); 
        subplot(3,4,10); imshow(squeeze(Ct(:,:,vis_s)),[]); 
        subplot(3,4,11); imshow(squeeze(I(:,:,vis_s)),[]); 
        subplot(3,4,12); imshow(squeeze(I2(:,:,vis_s)),[]); 
        
        disp(['Labelling error between implementations = ', num2str(sum(sum(sum(abs(I-I2)))))]);
        
        % convergence plots
        figure(); 
        subplot(1,2,1); loglog(erriter); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence mex/C');
        subplot(1,2,2); loglog(erriter2); xlim([1 maxIter]); ylim([min([erriter; erriter2]), max([erriter; erriter2])]); title('convergence Matlab/CUDA');
        
    end
    colormap('gray');
end



if(run2DBinaryMF_starShapeTestFLAG)
    
    % alloc a cost function Ct for each label i, int lId
    Ct = zeros(r,c, numberOfLabels);
    alpha = zeros(r,c, numberOfLabels);
    
    % for each label assign a point the star shape is enforced towards
    for i=1:numberOfLabels
        ss_initPoints(i,:) = [randi([1 c]), randi([1 r])];
    end
    
    % for each label assign a random data cost
    for i=1:numberOfLabels
        rng shuffle;
        Ct(:,:,i) = rand(r,c);
        h = fspecial('gaussian', [1 17], 1);
        Ct(:,:,i) = imfilter(Ct(:,:,i),h);
        
    end
    
    % for each label assign a constant regularization weight
    for i=1:numberOfLabels
        alpha(:,:,i) = (0.25).*ones(r,c);
    end
        
    % call max-flow optimizer
    % pars = [rows; columns; numberOfLabels; maxIter; convRate; cc; stepSize_s, stepSize_v];
    pars = [r; c; numberOfLabels; 1000; 1e-11; 0.2; 0.16; 0.7];
    
    % run both 2D matlab and mex implementations
    [u, erriter, i, timet] = asetsBinaryMF2D_starShape(Ct, alpha, pars, ss_initPoints);
    
    % maj vote to discretize continuous labels
    [um,I] = max(u, [], 3);
    
    % visualize
    if (visualizationFLAG)
        
        figure();
        for i=1:(numberOfLabels)
            subplot(2,numberOfLabels,i); imshow(Ct(:,:,i),[]);
            subplot(2,numberOfLabels,i+numberOfLabels); imshow(u(:,:,i),[0 1]);
        end
        
        % view resulting labeling functions and init points
        figure();
        subplot(1,2,1); imshow(I,[1 numberOfLabels]); hold on;
        colormap('jet');
        for i=1:numberOfLabels
            plot(ss_initPoints(i,1),ss_initPoints(i,2),'ob', 'MarkerSize', 10, 'MarkerFaceColor','w');
            text(ss_initPoints(i,1),ss_initPoints(i,2),num2str(i),...
            'FontSize',8,...
            'HorizontalAlignment','center');
        end
        colorbar();
        % plot convergence rate
        subplot(1,2,2); loglog(erriter);
        
        drawnow();
    end
    keyboard;
    
end


end




