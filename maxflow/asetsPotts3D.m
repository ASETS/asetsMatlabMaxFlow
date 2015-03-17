function [u, erriter, i, timet] = asetsPotts3D(Ct, alpha, pars)

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
slices = pars(3);
nlab = pars(4);
iterNum= pars(5);
beta = pars(6);
cc = pars(7);
steps = pars(8);


vol = rows*cols*slices*nlab;

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

u = zeros(rows,cols,slices,nlab, class(Ct));
pt = zeros(rows,cols,slices,nlab, class(Ct));
ps = zeros(rows,cols,slices, class(Ct));

% [ps,I] = min(Ct, [], 4);
% 
% for k=1:rows
%     for j=1:cols
%         pt(k,j,:) = ps(k,j);
%         u(k,j,I(k,j)) = 1;
%     end
% end

divp = zeros(rows,cols,slices,nlab, class(Ct));

pp1 = zeros(rows, cols+1, slices,nlab, class(Ct));
pp2 = zeros(rows+1, cols, slices,nlab, class(Ct));
pp3 = zeros(rows, cols, slices+1,nlab, class(Ct));

erriter = zeros(iterNum,1, class(Ct));

tic
for i = 1:iterNum
    
    pd = zeros(rows,cols,slices, class(Ct));
    
    % update the flow fields within each layer i=1...nlab
    
    for k= 1:nlab
        
        % update the spatial flow field p(x,i) = (pp1(x,i), pp2(x,i)):
        % the following steps are the gradient descent step with steps as the
        % step-size.
        
        ud = divp(:,:,:,k) - (ps - pt(:,:,:,k) + u(:,:,:,k)/cc);
        
        pp1(:,2:cols,:,k) = steps*(ud(:,2:cols,:) - ud(:,1:cols-1,:)) + pp1(:,2:cols,:,k);
        pp2(2:rows,:,:,k) = steps*(ud(2:rows,:,:) - ud(1:rows-1,:,:)) + pp2(2:rows,:,:,k);
        pp3(:,:,2:slices,k) = steps*(ud(:,:,2:slices) - ud(:,:,1:slices-1)) + pp3(:,:,2:slices,k);
        
        % the following steps are the projection to make |p(x,i)| <= alpha(x)
        
        gk = sqrt((pp1(:,1:cols,:,k).^2 + pp1(:,2:cols+1,:,k).^2 +...
            pp2(1:rows,:,:,k).^2 + pp2(2:rows+1,:,:,k).^2 + ...
                pp3(:,:,1:slices,k).^2 + pp3(:,:,2:slices+1,k).^2)*0.5);
        
        gk = double(gk <= alpha(:,:,:,k)) + double(~(gk <= alpha(:,:,:,k))).*(gk ./ alpha(:,:,:,k));
        gk = 1 ./ gk;
        
        pp1(:,2:cols,:,k) = (0.5*(gk(:,2:cols,:) + gk(:,1:cols-1,:))).*pp1(:,2:cols,:,k);
        pp2(2:rows,:,:,k) = (0.5*(gk(2:rows,:,:) + gk(1:rows-1,:,:))).*pp2(2:rows,:,:,k);
        pp3(:,:,2:slices,k) = (0.5*(gk(:,:,2:slices) + gk(:,:,1:slices-1))).*pp3(:,:,2:slices,k);
        
        divp(:,:,:,k) = pp1(:,2:cols+1,:,k)-pp1(:,1:cols,:,k) + ...
            pp2(2:rows+1,:,:,k)-pp2(1:rows,:,:,k) + ...
            pp3(:,:,2:slices+1,k)-pp3(:,:,1:slices,k);
        
        % update the sink flow field pt(x,i)
        
        ud = - divp(:,:,:,k) + ps + u(:,:,:,k)/cc;
        pt(:,:,:,k) = min(ud, Ct(:,:,:,k));
        
        % pd: the sum-up field for the computation of the source flow field
        % ps(x)
        %      
        
        pd = pd + (divp(:,:,:,k) + pt(:,:,:,k) - u(:,:,:,k)/cc);
        
    end
    
    % updata the source flow ps
    
    ps = pd / nlab + 1 / (cc*nlab);
    
    % update the multiplier u
    
    erru_sum = 0;
    for k = 1:nlab
        erru = cc*(divp(:,:,:,k) + pt(:,:,:,k) - ps);
        u(:,:,:,k) = u(:,:,:,k) - erru;
        erru_sum = erru_sum + sum(sum(sum(abs(erru))));
    end
    
    % evaluate the avarage error
    
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

msg = sprintf('number of iterations = %u; time = %.2f\n', i, timet);
disp(msg);


end