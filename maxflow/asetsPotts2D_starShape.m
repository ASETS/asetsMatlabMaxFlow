function [u, erriter, i, timet] = asetsPotts2D_starShape(Ct, alpha, pars, ss_initPoints)

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
stepv = pars(8);


vol = rows*cols*nlab;


for k=1:nlab
    % set hard constraints for init points on Ct
    Ct(ss_initPoints(k,2),ss_initPoints(k,1),:) = 10e11;
    Ct(ss_initPoints(k,2),ss_initPoints(k,1),k) = 0;
end

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

u = zeros(rows,cols,nlab);
pt = zeros(rows,cols,nlab);
%ps = zeros(rows,cols);



[ps,I] = min(Ct, [], 3);

% for k=1:rows
%     for j=1:cols
%         pt(k,j,:) = ps(k,j);
%         u(k,j,I(k,j)) = 1;
%     end
% end



divp = zeros(rows,cols,nlab);

pp1 = zeros(rows, cols+1,nlab);
pp2 = zeros(rows+1, cols,nlab);

% alloc the star shape multiplier v and other buffers
v = zeros(rows,cols,nlab);
vx1 = zeros(rows, cols+1,nlab);
vx2 = zeros(rows+1, cols,nlab);
gdv = zeros(rows, cols,nlab);

% compute the outward direction field
ve1 = zeros(rows,cols+1,nlab);
ve2 = zeros(rows+1,cols,nlab);
div_ve = zeros(rows,cols,nlab);

for k=1:nlab
    % compute the distance from the star shape init point
    tmp = zeros(rows,cols);
    tmp(ss_initPoints(k,2),ss_initPoints(k,1)) = 1;
    vd = bwdist(tmp,'euclidean');
    
    ve1(:,2:cols,k) = vd(:,2:cols) - vd(:,1:cols-1);
    ve2(2:rows,:,k) = vd(2:rows,:) - vd(1:rows-1,:);
    
    div_ve(:,:,k) = ve1(:,2:cols+1,k) - ve1(:,1:cols,k) + ...
        ve2(2:rows+1,:,k) - ve2(1:rows,:,k);
    clear tmp; clear vd;
end

erriter = zeros(iterNum,1);

tic
for i = 1:iterNum
    
    pd = zeros(rows,cols);
    
    % update the flow fields within each layer i=1...nlab
    
    for k= 1:nlab
        
        % update the spatial flow field p(x,i) = (pp1(x,i), pp2(x,i)):
        % the following steps are the gradient descent step with steps as the
        % step-size.
        
        ud = divp(:,:,k) + gdv(:,:,k) - (ps - pt(:,:,k) + u(:,:,k)/cc);
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
        
        ud = - (divp(:,:,k) + gdv(:,:,k)) + ps + u(:,:,k)/cc;
        pt(:,:,k) = min(ud, Ct(:,:,k));
        
        % pd: the sum-up field for the computation of the source flow field
        %      ps(x)
        
        pd = pd + (divp(:,:,k) + gdv(:,:,k) + pt(:,:,k) - u(:,:,k)/cc);
        
    end
    
    % update the source flow ps
    
    ps = pd / nlab + 1 / (cc*nlab);
    
    for k= 1:nlab
        % update v
        ud = gdv(:,:,k) + divp(:,:,k) + pt(:,:,k) - (ps + u(:,:,k)/cc);
        
        vx1(:,2:cols,k) = ud(:,2:cols) - ud(:,1:cols-1);
        vx2(2:rows,:,k) = ud(2:rows,:) - ud(1:rows-1,:);
        
        gk = (vx1(:,1:cols,k).*ve1(:,1:cols,k) + ...
            vx1(:,2:cols+1,k).*ve1(:,2:cols+1,k) + ...
            vx2(1:rows,:,k).*ve2(1:rows,:,k) + ...
            vx2(2:rows+1,:,k).*ve2(2:rows+1,:,k))*0.5;
        
        v(:,:,k) = v(:,:,k) + stepv*gk;
        v(:,:,k) = min(v(:,:,k),0);
        
        % update the div (v . p)
        vx1(:,2:cols,k) = v(:,2:cols,k) - v(:,1:cols-1,k);
        vx2(2:rows,:,k) = v(2:rows,:,k) - v(1:rows-1,:,k);
        
        gdv(:,:,k) = (vx1(:,1:cols,k).*ve1(:,1:cols,k) + ...
            vx1(:,2:cols+1,k).*ve1(:,2:cols+1,k) + ...
            vx2(1:rows,:,k).*ve2(1:rows,:,k) + ...
            vx2(2:rows+1,:,k).*ve2(2:rows+1,:,k))*0.5 + ...
            v(:,:,k) .* div_ve(:,:,k);
        
    end
    
    % update the multiplier u
    
    erru_sum = 0;
    for k = 1:nlab
        erru = cc*(divp(:,:,k) + gdv(:,:,k) + pt(:,:,k) - ps);
        u(:,:,k) = u(:,:,k) - erru;
        erru_sum = erru_sum + sum(sum(abs(erru)));
    end
    
    % evaluate the avarage error
    
    erriter(i) = erru_sum/vol;
    
    if erriter(i) < beta
        break;
    end
    
end

toc
timet = toc

msg = sprintf('number of iterations = %u. \n', i);
disp(msg);


end