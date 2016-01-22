classdef asetsDAGMF2D < handle
%   John SH Baxter, Robarts Research Institute, 2015
%
%   Full implementation with of [1] in 3D and DAGMF implementation
%   of [2]
%
%   [1] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM. (2014)
%       A Continuous Max-Flow Approach to Multi-Labeling Problems Under
%       Arbitrary Region Regularization
%       arXiv preprint arXiv:1404.0336
%
%   [2] Baxter, JSH.; Yuan, J.; Peters, TM.
%       Shape Complexes in Continuous Max-Flow Hierarchical Multi-
%       Labeling Problems
%       arXiv preprint arXiv:1510.04706
    
properties
    Name
    Ct
    alpha
    D
    
    u
    
    pn
    pt
    px
    py
    lx
    ly
    div
    g

    wC
    P
    C
end

methods

    %constructor
    function h = asetsDAGMF2D(children,alpha,Ct)
        %add in children
        h.C = children;
        for i = 1:length(children)
            h.C{i}.P{end+1} = h;
        end
        
        %find dimensions
        if isempty(children)
            h.D = size(Ct);
            h.Ct = Ct;
        else
            h.D = h.C{1}.D;
            h.Ct = zeros(0,'like',h.C{1}.Ct);
        end
        
        %create buffers
        h.alpha = alpha;
        h.u = zeros(h.D,'like',h.Ct);
        
    end
    
    %run full-flow algorithm with set number of iterations
    function MaxFullFlow(h,numIts,steps,cc)
        h.InitializeFullFlow();
        
        %find forward and backward orderings using a topological sort
        S = {h};
        tdList = {};
        while ~isempty(S)
            q = S{end};
            tdList{end+1} = q;
            S(end) = [];
            
            for j = 1:length(q.C)
                r = q.C{j};
                putInList = true;
                for k = 1:length(r.P)
                    s = r.P{k};
                    inList = false;
                    for l = 1:length(tdList)
                        if s == tdList{l}
                            inList = true;
                            break
                        end
                    end
                    if inList == false
                        putInList = false;
                        break;
                    end
                end
                if putInList
                    S{end+1} = r;
                end
            end
        end
        clear S;
        buList = flip(tdList);
        
        %run max-flow algorithm
        for i = 1:numIts
            
            %update spatial flows then clear input flow
            for j = 1:length(tdList)
                tdList{j}.UpdateSpatialFlows(steps,cc);
                if( ~isempty(tdList{j}.pn) )
                    tdList{j}.pn = zeros(tdList{j}.D,'like',tdList{j}.Ct);
                end
            end
            
            %push down sink (output) flows
            for j = 1:length(tdList)
                tdList{j}.PushDownFlows(cc);
            end
            
            %push up flow expectations/capacities
            for j = 1:length(buList)
                buList{j}.PushCapacityUp(cc);
            end
            
            %update labels
            for j = 1:length(tdList)
                tdList{j}.UpdateLabels(cc);
            end
            drawnow
        end
        
        h.DeInitializeFullFlow();
    end
    
    %initialization procedure for full flow including
    %default parameterization, normalization and buffers    
    function InitializeFullFlow(h)
        
        %give default weights
        if isempty(h.wC)
            for i = 1:length(h.C)
                h.wC(i) = 1/length(h.C{i}.P);
            end
        end
        
        %initialize buffers for full flow
        for i = 1:length(h.C)
            h.C{i}.InitializeFullFlow();
        end
        h.g = zeros(h.D,'like',h.Ct);
        h.pt = zeros(h.D,'like',h.Ct);
        if( length(h.P) > 1 )
            h.pn = zeros(h.D,'like',h.Ct);
        end
        h.div = zeros(h.D,'like',h.Ct);
        h.u = zeros(h.D,'like',h.Ct);
        h.px = zeros([h.D(1)-1 h.D(2)],'like',h.Ct);
        h.py = zeros([h.D(1) h.D(2)-1],'like',h.Ct);
        
        %normalize lengths for geodesic shape constraint
        if numel(h.lx) > 0 && numel(h.ly) > 0
            denom = (h.lx.^2+h.ly.^2).^0.5;
            mask = (denom > 0.001);
            h.lx(mask) =  h.lx(mask) ./ denom(mask);
            h.lx(~mask) = 0;
            h.ly(mask) =  h.ly(mask) ./ denom(mask);
            h.ly(~mask) = 0;
        end
        
    end
    
    %deinitialize buffers for full flow
    function DeInitializeFullFlow(h)
        for i = 1:length(h.C)
            h.C{i}.DeInitializeFullFlow();
        end
        clear h.g;
        clear h.pt;
        clear h.pn;
        clear h.px;
        clear h.py;
        clear h.div;
    end
    
    
    %update labels (top-down) (not recursive)
    function UpdateLabels(h,cc)
        if length(h.P)>1
            h.u = h.u + cc*(h.pn - h.div - h.pt);
        elseif length(h.P)==1
            h.u = h.u + cc*(h.P{1}.pt - h.div - h.pt);
        end
        
        if ~isempty(h.Name)
            PlotResults(h.Name,h.px,h.py,h.pt,h.div,h.g,h.u)
        end
    end

    %push down sink flow step (not recursive)
    function PushDownFlows(h,cc)
        %push down flow
        for i = 1:length(h.C)
            if length(h.C{i}.P)>1
                h.C{i}.pn = h.C{i}.pn + h.wC(i)*h.pt;
            end
        end
        
        %figure out sink desired capacity
        if ~isempty(h.P) && ~isempty(h.C)
            if length(h.P)>1
                h.g = h.pn - h.div + h.u/cc;
            else
                h.g = h.P{1}.pt - h.div + h.u/cc;
            end
        elseif isempty(h.P)
            h.g = 1/cc;
        end
    end
    
    %push sink capacities up and set sink flows
    function PushCapacityUp(h,cc)
        %leaf node, so constrain output flow
        if isempty(h.C)
            if length(h.P) > 1
                h.pt = min(h.Ct, h.pn-h.div+h.u/cc);
            else
                h.pt = min(h.Ct, h.P{1}.pt-h.div+h.u/cc);
            end
            
        %source node, so accumulate desired in-flow from children and set
        %source flow accordingly
        elseif isempty(h.P)
            for i = 1:length(h.C)
                if length(h.C{i}.P)>1
                    h.g = h.g + h.wC(i)*(h.C{i}.div + h.C{i}.pt + h.wC(i)*h.pt - h.C{i}.pn - h.C{i}.u/cc);
                else
                    h.g = h.g + h.C{i}.div + h.C{i}.pt - h.C{i}.u/cc;
                end
            end
            h.pt = h.g / sum(h.wC .^2);
            
        %branch node, so accumulate desired in-flow from children and set
        %branch output flow accordingly
        else
            for i = 1:length(h.C)
                if length(h.C{i}.P)>1
                    h.g = h.g + h.wC(i)*(h.C{i}.div + h.C{i}.pt + h.wC(i)*h.pt - h.C{i}.pn - h.C{i}.u/cc);
                else
                    h.g = h.g + h.C{i}.div + h.C{i}.pt - h.C{i}.u/cc;
                end
            end
            h.pt = h.g / (1+sum(h.wC .^2));
        end
        
    end
    
    %update spatial flows (not recursive)
    function UpdateSpatialFlows(h,steps,cc)
        if ~isempty(h.P)

            %gradient descent on flows
            if length(h.P) > 1
                h.g = steps*( h.div + h.pt - h.pn - h.u/cc );
            else
                h.g = steps*( h.div + h.pt - h.P{1}.pt - h.u/cc );
            end
            h.px = h.px + h.g(2:h.D(1),:)-h.g(1:h.D(1)-1,:);
            h.py = h.py + h.g(:,2:h.D(2))-h.g(:,1:h.D(2)-1);

            %find flow mag, exemption amounts, and correction
            if numel(h.lx) > 0
                
                %find exemption amount
                h.g(1:h.D(1)-1,:) =                  max((h.px>0).*h.px.*h.lx(1:h.D(1)-1,:),0);
                h.g(h.D(1),:)=0;
                h.g(2:h.D(1),:)   = h.g(2:h.D(1),:)  + max((h.px<0).*h.px.*h.lx(2:h.D(1),:)  ,0);
                h.g(:,1:h.D(2)-1) = h.g(:,1:h.D(2)-1)+ max((h.py>0).*h.py.*h.ly(:,1:h.D(2)-1),0);
                h.g(:,2:h.D(2))   = h.g(:,2:h.D(2))  + max((h.py<0).*h.py.*h.ly(:,2:h.D(2))  ,0);
                
                %find exemption amount
                ex =      (h.px>0 & h.lx(1:h.D(1)-1,:,:)>0).*h.lx(1:h.D(1)-1,:,:).*h.g(1:h.D(1)-1,:,:);
                ex = ex + (h.px<0 & h.lx(2:h.D(1),:,:)<0).*h.lx(2:h.D(1),:,:).*h.g(2:h.D(1),:,:);
                
                ey =      (h.py>0 & h.ly(:,1:h.D(2)-1,:)>0).*h.ly(:,1:h.D(2)-1,:).*h.g(:,1:h.D(2)-1,:);
                ey = ey + (h.py<0 & h.ly(:,2:h.D(2),:)<0).*h.ly(:,2:h.D(2),:).*h.g(:,2:h.D(2),:);
                
                %apply exemption
                h.px = h.px - ex;
                h.py = h.py - ey;
                
                %find flow mag
                h.g(1:h.D(1)-1,:) = h.px.^2;
                h.g(h.D(1),:)=0;
                h.g(2:h.D(1),:)   = h.g(2:h.D(1),:)   + h.px.^2;
                h.g(:,1:h.D(2)-1) = h.g(:,1:h.D(2)-1) + h.py.^2;
                h.g(:,2:h.D(2))   = h.g(:,2:h.D(2))   + h.py.^2;
                h.g = h.g .^ 0.5;
                
                %correct for flow mag
                mask = (h.g <= h.alpha);
                if numel(h.alpha) == 1
                    h.g(~mask) = h.alpha ./ h.g(~mask);
                else
                    h.g(~mask) = h.alpha(~mask) ./ h.g(~mask);
                end
                h.g(mask) = 1;
                h.px = ex + 0.5 * h.px .* (h.g(2:h.D(1),:)+h.g(1:h.D(1)-1,:));
                h.py = ey + 0.5 * h.py .* (h.g(:,2:h.D(2))+h.g(:,1:h.D(2)-1));
                
            %no exemption vector, so a=0
            else

                %find flow mag
                h.g(1:h.D(1)-1,:) = h.px.^2;
                h.g(h.D(1),:)=0;
                h.g(2:h.D(1),:)   = h.g(2:h.D(1),:)   + h.px.^2;
                h.g(:,1:h.D(2)-1) = h.g(:,1:h.D(2)-1) + h.py.^2;
                h.g(:,2:h.D(2))   = h.g(:,2:h.D(2))   + h.py.^2;
                h.g = h.g .^ 0.5;

                %correct for flow mag
                mask = (h.g <= h.alpha);
                if numel(h.alpha) == 1
                    h.g(~mask) = h.alpha ./ h.g(~mask);
                else
                    h.g(~mask) = h.alpha(~mask) ./ h.g(~mask);
                end
                h.g(mask) = 1;
                h.px = 0.5 * h.px .* (h.g(2:h.D(1),:)+h.g(1:h.D(1)-1,:));
                h.py = 0.5 * h.py .* (h.g(:,2:h.D(2))+h.g(:,1:h.D(2)-1));
            end
            
            %calculate divergence
            h.div(1:h.D(1)-1,:) = h.px;
            h.div(h.D(1),:) = 0;
            h.div(2:h.D(1),:)   = h.div(2:h.D(1),:)   - h.px;
            h.div(:,1:h.D(2)-1) = h.div(:,1:h.D(2)-1) + h.py;
            h.div(:,2:h.D(2))   = h.div(:,2:h.D(2))   - h.py;
            
        end
        
    end
    
    
end

end