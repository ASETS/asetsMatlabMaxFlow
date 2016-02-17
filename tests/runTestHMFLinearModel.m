function [] = runTestHMFLinearModel()

%   John Baxter, Robarts Research Institute, London, Ontario, 2015
%   Testing scripts for hierarchical max flow regularization
%   Test script comprises of a linear (Ishikawa) image reconstruction model

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

alpha = 0.15;
noise = 0.15;
showVis = 0;

%create image to denoise
image = mat2gray(imread('cell.tif'));
imageDenoised = image;
image = image + noise*randn(size(image));

%create max-flow model
numLevels = 40;
endlabel = cell(1,numLevels);
for i = 1:numLevels
    endlabel{i} = asetsHMF2D({},alpha,sqrt(abs(image-(i-1)/(numLevels-1))));
end
s = asetsHMF2D(endlabel,0);
tic; s.MaxFullFlow(200,0.1,1/numLevels); toc

%reconstruct image
accum = zeros(size(image));
for i = 1:numLevels
    accum = accum + endlabel{i}.u;
end
recon = zeros(size(image));
for i = 1:numLevels
    recon = recon + endlabel{i}.u .* ((i-1)/(numLevels-1));
end
recon = recon ./ accum;

%get norms for display
maxTFlow = max(s.pt(:));
maxCFlow = 0;
maxSFlow = 0;
for i = 1:numLevels
    maxCFlow = max(maxTFlow,max(endlabel{i}.Ct(:)));
    maxTFlow = max(maxTFlow,max(endlabel{i}.pt(:)));
    maxSFlow = max( maxSFlow, ...
                    max(max(endlabel{i}.px(:)), -min(endlabel{i}.px(:))) );
end

figure(2)
    subplot(4,1,1);
        imshow(image(:,:,1),[0,1]); title('Noisy image');
    subplot(4,1,2);
        imshow(recon(:,:,1),[0,1]); title('Reconstructed image');
    subplot(4,1,3);
        imshow(abs(recon(:,:,1)-imageDenoised(:,:,1)),[0 1]); 
    subplot(4,1,4);
        imshow(abs(recon(:,:,1)-image(:,:,1)),[0 1]); 

if showVis 
    figure(1); clf;
        subplot(7,1+numLevels,2*(numLevels+1)+1);
            imshow(s.pt,[0,maxTFlow]);
    for i = 1:numLevels
        subplot(7,1+numLevels,0*(numLevels+1)+1+i);
            imshow(endlabel{i}.u,[0,1]);
        subplot(7,1+numLevels,1*(numLevels+1)+1+i);
            imshow(endlabel{i}.Ct,[0,maxCFlow]);
        subplot(7,1+numLevels,2*(numLevels+1)+1+i);
            imshow(endlabel{i}.pt,[0,maxTFlow]);
        subplot(7,1+numLevels,3*(numLevels+1)+1+i);
            imshow(endlabel{i}.div,[-maxSFlow,maxSFlow]);
        subplot(7,1+numLevels,4*(numLevels+1)+1+i);
            imshow(endlabel{i}.px,[-maxSFlow,maxSFlow]);
        subplot(7,1+numLevels,5*(numLevels+1)+1+i);
            imshow(endlabel{i}.py,[-maxSFlow,maxSFlow]);
        subplot(7,1+numLevels,6*(numLevels+1)+1+i);
            imshow(endlabel{i}.g,[0 1]);
    end
end

end