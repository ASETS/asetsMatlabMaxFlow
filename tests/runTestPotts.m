function [] = runTestPotts()

%   Martin Rajchl, Imperial College London, 2015
%   Testing scripts for Potts model regularization

close all;
clear all;

addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% flags
run2DPottsTestFLAG = 1;
run3DPottsTestFLAG = 1;

run2DPotts_starShapeTestFLAG = 0;

visualizationFLAG = 1;

% parameters
numberOfLabels = 10;
r = 64; % number of rows
c = 64; % number of columns
s = 32; % number of slices

if (run2DPottsTestFLAG)
    
    % alloc a cost function Ct for each label i, int lId
    Ct = zeros(r,c, numberOfLabels);
    alpha = zeros(r,c, numberOfLabels);
    
    % for each label assign a random data cost
    for i=1:numberOfLabels
        rng shuffle;
        Ct(:,:,i) = rand(r,c);
        h = fspecial('gaussian', [1 17], 1);
        Ct(:,:,i) = imfilter(Ct(:,:,i),h);
    end
    
    % for each label assign a constant regularization weight
    for i=1:numberOfLabels
        alpha(:,:,i) = (0.05*i).*ones(r,c);
    end
    
    % call max-flow optimizer
    % pars = [rows; columns; numberOfLabels; maxIter; convRate; cc; stepSize];
    pars = [r; c; numberOfLabels; 200; 1e-11; 0.2; 0.16];
    
    % run both 2D matlab and mex implementations
    [u, erriter, i, timet] = asetsPotts2D_mex(single(Ct), single(alpha), single(pars));
    [u2, erriter2, i2, timet2] = asetsPotts2D(Ct, alpha, pars);
    
    
    % maj vote to discretize continuous labels
    [um,I] = max(u, [], 3);
    [um,I2] = max(u2, [], 3);
    
    % visualize
    if (visualizationFLAG)
        
        figure();
        for i=1:(numberOfLabels)
            subplot(2,numberOfLabels,i); imshow(Ct(:,:,i),[]);
            subplot(2,numberOfLabels,i+numberOfLabels); imshow(u(:,:,i),[0 1]);
        end
        
        % view resulting labeling functions from each implementation
        figure();
        subplot(1,2,1); imshow(I,[1 numberOfLabels]);
        subplot(1,2,2); imshow(I2,[1 numberOfLabels]);
        
        disp(['Labelling error between implementations = ', num2str(sum(sum(abs(I-I2))))]);
    end
    colormap('jet');
    
end



if (run3DPottsTestFLAG)
    
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
        alpha(:,:,:,i) = (0.05*i).*ones(r,c,s);
    end
    
    % call 3D max-flow optimizer
    
    % pars = [rows; columns; slices; numberOfLabels; maxIter; convRate; cc; stepSize];
    pars = [r; c; s; numberOfLabels; 200; 1e-11; 0.25; 0.11];
    
    % run both 3D matlab and mex implementations
    [u, erriter, i, timet] = asetsPotts3D_mex(single(Ct), single(alpha), single(pars));
    [u2, erriter2, i2, timet2] = asetsPotts3D(Ct, alpha, pars);
    
    
    % maj vote to discretize continuous labels
    [um,I] = max(u, [], 4);
    [um,I2] = max(u2, [], 4);
    
    % visualize
    if(visualizationFLAG)
        
        % compute mid slices in each direction
        vis_r = idivide(r,uint8(2));
        vis_c = idivide(c,uint8(2));
        vis_s = idivide(s,uint8(2));
        
        figure();
        for i=1:(numberOfLabels)
            subplot(4,numberOfLabels,i); imshow(Ct(:,:,vis_s,i),[]);
            subplot(4,numberOfLabels,i+numberOfLabels); imshow(squeeze(u(vis_r,:,:,i)),[0 1]);
            subplot(4,numberOfLabels,i+2*numberOfLabels); imshow(squeeze(u(:,vis_c,:,i)),[0 1]);
            subplot(4,numberOfLabels,i+3*numberOfLabels); imshow(squeeze(u(:,:,vis_s,i)),[0 1]);
        end
        
        % view resulting labeling functions from each implementation
        figure();
        subplot(2,3,1); imshow(squeeze(I(vis_r,:,:)),[1 numberOfLabels]);
        subplot(2,3,2); imshow(squeeze(I(:,vis_c,:)),[1 numberOfLabels]);
        subplot(2,3,3); imshow(squeeze(I(:,:,vis_s)),[1 numberOfLabels]);
        subplot(2,3,4); imshow(squeeze(I2(vis_r,:,:)),[1 numberOfLabels]);
        subplot(2,3,5); imshow(squeeze(I2(:,vis_c,:)),[1 numberOfLabels]);
        subplot(2,3,6); imshow(squeeze(I2(:,:,vis_s)),[1 numberOfLabels]);
        
        colormap('jet');
        disp(['Labeling error between implementations = ', num2str(sum(sum(sum(abs(I-I2)))))]);
        
    end
end

if(run2DPotts_starShapeTestFLAG)
    
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
    [u, erriter, i, timet] = asetsPotts2D_starShape(Ct, alpha, pars, ss_initPoints);
    
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




