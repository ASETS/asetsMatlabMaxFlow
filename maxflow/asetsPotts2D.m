function [u, erriter, i, timet] = asetsPotts2D(Ct, alpha, pars)

%   Martin Rajchl, Imperial College London, 2015
%
%   Re-implementation with of [1] with a horizontal model
%   according to [2]
%
%   [1] Yuan, J.; Bae, E.; Tai, X.-C.; Boykov, Y.
%       A Continuous Max-Flow Approach to Potts Model
%       ECCV, 2010
%
%   [2] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Continuous Max-Flow Approach to General
%       Hierarchical Multi-Labelling Problems
%       arXiv preprint arXiv:1404.0336


if(nargin < 3)
    error('Not enough args. Exiting...');
end

% setup
rows = pars(1);
cols = pars(2);
nlab = pars(3);
iterNum= pars(4);
beta = pars(5);
cc = pars(6);
steps = pars(7);

vol = rows*cols*nlab;

% set the initial values:
%   - u(x,i=1...nlab) is set to be an initial cut, see below.
%
%   - the source flow field ps(x), see below.
%
%   - the nlab sink flow fields pt(x,i=1...nlab), set to be the specified
%     legal flows.
%
%   - the spatial flow fiels p(x,i=1...nlab) = (pp1(x,i), pp2(x,i)),
%     set to be zero.

u = zeros(rows,cols,nlab, 'like', Ct);
pt = zeros(rows,cols,nlab, 'like', Ct);
ps = zeros(rows,cols, 'like', Ct);

% initialize the flow buffers for faster convergence
[ps,I] = min(Ct, [], 3);

for i=1:nlab
    pt(:,:,i) = ps;
    tmp = I == i;
    u(:,:,i) = tmp;
end

divp = zeros(rows,cols,nlab, 'like', Ct);

pp1 = zeros(rows, cols+1,nlab, 'like', Ct);
pp2 = zeros(rows+1, cols,nlab, 'like', Ct);

erriter = zeros(iterNum,1, 'like', Ct);

tic
for i = 1:iterNum
    
    pd = zeros(rows,cols, 'like', Ct);
    
    % update the flow fields within each layer i=1...nlab
    
    for k= 1:nlab
        
        % update the spatial flow field p(x,i) = (pp1(x,i), pp2(x,i)):
        % the following steps are the gradient descent step with steps as the
        % step-size.
        
        ud = divp(:,:,k) - (ps - pt(:,:,k) + u(:,:,k)/cc);
        pp1(:,2:cols,k) = steps*(ud(:,2:cols) - ud(:,1:cols-1)) + pp1(:,2:cols,k);
        pp2(2:rows,:,k) = steps*(ud(2:rows,:) - ud(1:rows-1,:)) + pp2(2:rows,:,k);
        
        % the following steps are the projection to make |p(x,i)| <= alpha(x)
        
        gk = sqrt((pp1(:,1:cols,k).^2 + pp1(:,2:cols+1,k).^2 +...
            pp2(1:rows,:,k).^2 + pp2(2:rows+1,:,k).^2)*0.5);
        
        gk = double(gk <= alpha(:,:,k)) + double(~(gk <= alpha(:,:,k))).*(gk ./ alpha(:,:,k));
        gk = 1 ./ gk;
        
        pp1(:,2:cols,k) = (0.5*(gk(:,2:cols) + gk(:,1:cols-1))).*pp1(:,2:cols,k);
        pp2(2:rows,:,k) = (0.5*(gk(2:rows,:) + gk(1:rows-1,:))).*pp2(2:rows,:,k);
        
        divp(:,:,k) = pp1(:,2:cols+1,k)-pp1(:,1:cols,k)+pp2(2:rows+1,:,k)-pp2(1:rows,:,k);
        
        % update the sink flow field pt(x,i)
        
        ud = - divp(:,:,k) + ps + u(:,:,k)/cc;
        pt(:,:,k) = min(ud, Ct(:,:,k));
        
        % pd: the sum-up field for the computation of the source flow field
        %      ps(x)
        
        pd = pd + (divp(:,:,k) + pt(:,:,k) - u(:,:,k)/cc);
        
    end
    
    % update the source flow ps
    ps = pd / nlab + 1 / (cc*nlab);
    
    % update the multiplier u
    erru_sum = 0;
    for k = 1:nlab
        erru = cc*(divp(:,:,k) + pt(:,:,k) - ps);
        u(:,:,k) = u(:,:,k) - erru;
        erru_sum = erru_sum + sum(sum(abs(erru)));
    end
    
    % evaluate the average error  
    erriter(i) = erru_sum/vol;
    
    if erriter(i) < beta
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