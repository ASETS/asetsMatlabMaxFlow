function [u, erriter, i, timet] = asetsBinaryMF2D(Cs, Ct, alpha, pars)

%   Martin Rajchl, Imperial College London, 2015
%
%   Re-implementation with of [1] with CUDA capability as in [2]
%
%   [1] Yuan, J.; Bae, E.; Tai, X,-C.;
%       A study on continuous max-flow and min-cut approaches
%       IEEE CVPR, 2010
%
%   [2] Rajchl, M.; Yuan, J.; Peters, TM.
%       Real-time segmentation in 4D ultrasound with 
%       continuous max-flow
%       SPIE Medical Imaging 2012, 83141F-83141F-8


if(nargin < 3)
    error('Not enough args. Exiting...');
end

% setup
rows = pars(1);
cols = pars(2);
iterNum= pars(3);
beta = pars(4);
cc = pars(5);
steps = pars(6);

imgSize = rows*cols;

% allocate buffers
u = zeros(rows,cols,class(Ct));
ps = zeros(rows,cols,class(Ct));
pt = zeros(rows,cols,class(Ct));

pp1 = zeros(rows, cols+1, class(Ct));
pp2 = zeros(rows+1, cols, class(Ct));
divp = zeros(rows,cols, class(Ct));

erriter = zeros(iterNum,1, class(Ct));

% initialize the flow buffers for faster convergence
u = double((Cs-Ct) >= 0);
ps = min(Cs, Ct);
pt = ps;

tic
for i = 1:iterNum
    
    % update the spatial flow field p = (pp1, pp2):
    % compute the gradient descent
    pts = divp - (ps - pt  + u/cc);
    pp1(:,2:cols) = pp1(:,2:cols) + steps*(pts(:,2:cols) - pts(:,1:cols-1)); 
    pp2(2:rows,:) = pp2(2:rows,:) + steps*(pts(2:rows,:) - pts(1:rows-1,:));
    
    % projection step |p(x)| <= alpha(x)
    gk = sqrt((pp1(:,1:cols).^2 + pp1(:,2:cols+1).^2 + pp2(1:rows,:).^2 + pp2(2:rows+1,:).^2)*0.5);
    gk = double(gk <= alpha) + double(~(gk <= alpha)).*(gk ./ alpha);
    gk = 1 ./ gk;
    
    pp1(:,2:cols) = (0.5*(gk(:,2:cols) + gk(:,1:cols-1))).*pp1(:,2:cols); 
    pp2(2:rows,:) = (0.5*(gk(2:rows,:) + gk(1:rows-1,:))).*pp2(2:rows,:);
    
    divp = pp1(:,2:cols+1)-pp1(:,1:cols)+pp2(2:rows+1,:)-pp2(1:rows,:);
    
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
    
    erriter(i) = sum(sum(abs(erru)))/imgSize; 
   
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