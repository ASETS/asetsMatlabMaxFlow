function [u, erriter, i, timet] = asetsIshikawa3D(Ct, alpha, pars)

%   Martin Rajchl, Imperial College London, 2015
%
%   Re-implementation with of [1] 
%
%   [1] Rajchl M., J. Yuan, E. Ukwatta, and T. Peters (2012).
%       Fast Interactive Multi-Region Cardiac Segmentation With 
%       Linearly Ordered Labels. 
%       ISBI, 2012. pp.1409–1412.


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

vol = rows*cols*slices*nlab-1;

% alloc the max flow buffers
pt = zeros(rows, cols, slices, nlab, 'like', Ct);
u = ones(rows, cols, slices, nlab-1, 'like', Ct);
pp1 = zeros(rows, cols+1, slices, nlab-1, 'like', Ct);
pp2 = zeros(rows+1, cols, slices, nlab-1, 'like', Ct);
pp3 = zeros(rows, cols, slices+1, nlab-1, 'like', Ct);
erriter = zeros(iterNum,1, 'like', Ct);
divp = zeros(rows,cols, slices, nlab-1, 'like', Ct);

% initialize the flow buffers for faster convergence
[um,init] = min(Ct,[],4);

for k=1:nlab
    pt(:,:,:,k) = um;
end

for k=1:nlab-1 
    tmp = init > k;
    u(:,:,:,k) = tmp;
end

tic
for i = 1:iterNum
    
    % update flow within each layer
    for k= 1:nlab-1
          
        ud = divp(:,:,:,k) - (pt(:,:,:,k) - pt(:,:,:,k+1) + u(:,:,:,k)/cc);
        
         % the following steps are the gradient descent step with steps as the
        % step-size.        
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
    end
    for k =1:nlab
        if (k == 1)
            ud = divp(:,:,:,k) + pt(:,:,:,2) - u(:,:,:,1)/cc + 1/cc;
            pt(:,:,:,1) = min(ud, Ct(:,:,:,1));
        end
        
        if (k >= 2) & (k < nlab)
            ud = - divp(:,:,:,k-1) + pt(:,:,:,k-1) + u(:,:,:,k-1)/cc;
            ud = ud + divp(:,:,:,k) + pt(:,:,:,k+1) - u(:,:,:,k)/cc;
            ud = ud/2;
            
            pt(:,:,:,k) = min(ud,Ct(:,:,:,k));
        end
        
        if (k==nlab)
            ud = - divp(:,:,:,nlab-1) + pt(:,:,:,nlab-1) + u(:,:,:,nlab-1)/cc;
            pt(:,:,:,nlab) = min(ud, Ct(:,:,:,nlab));
        end
    end
    
    % update the multiplier u
    erru_sum = 0;
    for k = 1:nlab-1
        erru = cc*(divp(:,:,:,k) + pt(:,:,:,k+1) - pt(:,:,:,k));
        u(:,:,:,k) = u(:,:,:,k) - erru;
        erru_sum = erru_sum + sum(sum(sum(abs(erru))));
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


