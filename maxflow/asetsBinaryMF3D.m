function [u, erriter, i, timet] = asetsBinaryMF3D(Cs, Ct, alpha, pars)

%   Martin Rajchl, Imperial College London, 2015
%
%   Re-implementation with of [1] with CUDA capability as in [2]
%
%   [1] Yuan, J.; Bae, E.; Tai, X,-C.;
%       A study on continuous max-flow and min-cut approaches
%       IEEE CVPR, 2010
%


if(nargin < 3)
    error('Not enough args. Exiting...');
end

% setup
rows = pars(1);
cols = pars(2);
slices = pars(3);
iterNum= pars(4);
beta = pars(5);
cc = pars(6);
steps = pars(7);

imgSize = rows*cols*slices;

%u = double((Cs-Ct) >= 0);
%ps = min(Cs, Ct);
u = zeros(rows,cols,slices,class(Ct));
ps = zeros(rows,cols,slices,class(Ct));
pt = zeros(rows,cols,slices,class(Ct));

pp1 = zeros(rows, cols+1, slices, class(Ct));
pp2 = zeros(rows+1, cols, slices, class(Ct));
pp3 = zeros(rows, cols, slices+1, class(Ct));
divp = zeros(rows,cols,slices, class(Ct));

erriter = zeros(iterNum,1, class(Ct));

tic
for i = 1:iterNum
    
    % update the spatial flow field p = (pp1, pp2):
    % compute the gradient descent
    pts = divp - (ps - pt  + u/cc);
    pp1(:,2:cols,:) = pp1(:,2:cols,:) + steps*(pts(:,2:cols,:) - pts(:,1:cols-1,:)); 
    pp2(2:rows,:,:) = pp2(2:rows,:,:) + steps*(pts(2:rows,:,:) - pts(1:rows-1,:,:));
    pp3(:,:,2:slices) = pp3(:,:,2:slices) + steps*(pts(:,:,2:slices) - pts(:,:,1:slices-1));
    
    % projection step |p(x)| <= alpha(x)
    gk = sqrt((pp1(:,1:cols,:).^2 + pp1(:,2:cols+1,:).^2 ...
             + pp2(1:rows,:,:).^2 + pp2(2:rows+1,:,:).^2 ...
             + pp3(:,:,1:slices).^2 + pp3(:,:,2:slices+1).^2 ...
             )*0.5);
    gk = double(gk <= alpha) + double(~(gk <= alpha)).*(gk ./ alpha);
    gk = 1 ./ gk;
    
    pp1(:,2:cols,:) = (0.5*(gk(:,2:cols,:) + gk(:,1:cols-1,:))).*pp1(:,2:cols,:); 
    pp2(2:rows,:,:) = (0.5*(gk(2:rows,:,:) + gk(1:rows-1,:,:))).*pp2(2:rows,:,:);
    pp3(:,:,2:slices) = (0.5*(gk(:,:,2:slices) + gk(:,:,1:slices-1))).*pp3(:,:,2:slices);
    
    divp = pp1(:,2:cols+1,:)-pp1(:,1:cols,:) ...
         + pp2(2:rows+1,:,:)-pp2(1:rows,:,:) ...
         + pp3(:,:,2:slices+1) - pp3(:,:,1:slices);
    
    % update the source flow ps
    pts = divp + pt - u/cc + 1/cc;
    ps = min(pts, Cs);
    
    % update the sink flow pt
    pts = - divp + ps + u/cc;
    pt = min(pts, Ct);

	% update the multiplier u
    
	erru = cc*(divp + pt  - ps);
	u = u - erru;
    
    % evaluate the avarage error
    
    erriter(i) = sum(sum(sum(abs(erru))))/imgSize; 
   
    if (erriter(i) < beta)
        break;
    end
    
end

if(strcmp(class(u), 'gpuArray'))
    u = gather(u);
    erriter = gather(erriter);
end

timet = toc;

msg = sprintf('number of iterations = %u; time = %f \n', i, timet);
disp(msg);


end