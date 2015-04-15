%% Tutorial 03: Using different implentation of solvers (Matlab/C/CUDA)
%  Martin Rajchl, Imperial College London, 2015
%
%   [1] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Continuous Max-Flow Approach to General
%       Hierarchical Multi-Labelling Problems
%       arXiv preprint arXiv:1404.0336
%
%   [2] Rajchl, M.; Yuan, J.; Peters, TM.
%       Real-time segmentation in 4D ultrasound with 
%       continuous max-flow
%       SPIE Medical Imaging 2012, 83141F-83141F-8
  
clear all; close all;

% include max-flow solver
addpath(['..', filesep, 'maxflow']);
addpath(['..', filesep, 'lib']);

% 1. Load a volume image 
load(['..', filesep, 'data', filesep, 'brain_1125.mat'], 'img');

[r,c,s] = size(img);

% 2. Normalize the image intensity to [0,1]:
img = single(img);
img_n = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% 3. Create two cost functions as in Tutorial 01
val_fg = 0.75; val_bg = 0.25;

Cs = abs(img_n - val_fg);
Ct = abs(img_n - val_bg);

alpha = 0.05.*ones(r,c,s);

% 4. Setup execution parameters:
pars = [r; c; s; 200; 1e-11; 0.25; 0.11];

% 5. Run different implementations of the optimizer on the problem:

% Matlab with double precision
[u1, conv1, i1, time1] = asetsBinaryMF3D(double(Cs), double(Ct), double(alpha), pars);

% Matlab with single precision
[u2, conv2, i2, time2] = asetsBinaryMF3D(single(Cs), single(Ct), single(alpha), pars);

% MEX/C implementation with single precision
if(exist('../lib/asetsBinaryMF3D_mex','file'))
    [u3, conv3, i3, time3] = asetsBinaryMF3D_mex(single(Cs), single(Ct), single(alpha), single(pars));
else
     error('No compiled solvers found. Skipping computation. Please run the ./compile script and retry.');
end

% Matlab-interal CUDA implementation with single precision
if(gpuDeviceCount)
    [u4, conv4, i4, time4] = asetsBinaryMF3D(gpuArray(Cs), gpuArray(Ct), gpuArray(alpha), pars);
else
    error('No CUDA devices detected. Skipping computation.');
end

% threshold discretize continuous labels
ut1 = u1 > 0.5;
ut2 = u2 > 0.5;
ut3 = u3 > 0.5;
ut4 = u4 > 0.5;

% visualize the results
figure(); slice = 45;
subplot(2,5,1); imshow(img(:,:,slice),[]); title('Original image');
subplot(2,5,2); imshow(ut1(:,:,slice),[]); title(['Seg Matlab (single) in ', num2str(time1),'s.']);
subplot(2,5,3); imshow(ut2(:,:,slice),[]); title(['Seg Matlab (double) in ', num2str(time2),'s.']);
subplot(2,5,4); imshow(ut3(:,:,slice),[]); title(['Seg MEX/C (single) in ', num2str(time3),'s.']);
subplot(2,5,5); imshow(ut4(:,:,slice),[]); title(['Seg CUDA (single) in ', num2str(time4),'s.']);

subplot(2,5,7); loglog(conv1); title('Convergence Matlab (single)');
subplot(2,5,8); loglog(conv2); title('Convergence Matlab (double)');
subplot(2,5,9); loglog(conv3); title('Convergence MEX/C (single)');
subplot(2,5,10); loglog(conv4); title('Convergence CUDA (single)');
