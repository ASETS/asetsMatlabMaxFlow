classdef asetsHMF2D < handle
%   John SH Baxter, Robarts Research Institute, 2015
%
%   Full implementation with of [1,2] in 2D and HMF implementation
%   of [3]. Optional geodesic shape constraints (via lx & ly) are 
%   implemented as described in [4].
%
%   [1] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Continuous Max-Flow Approach to General
%       Hierarchical Multi-Labelling Problems
%       arXiv preprint arXiv:1404.0336
%
%   [2] Rajchl M., J. Baxter, A. McLeod, J. Yuan, W. Qiu, 
%       T. Peters, and A. Khan (2015). 
%       Hierarchical Max-Flow Segmentation Framework For Multi-Atlas 
%       Segmentation with Kohonen Self-Organizing Map Based Gaussian
%       Mixture Modeling. Medical Image Analysis
%
%   [3] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Proximal Bregman Projection Approach to Continuous
%       Max-Flow Problems Using Entropic Distances
%       arXiv preprint arXiv:1501.07844
%
%   [4] Baxter, JSH.; Yuan, J.; Peters, TM.
%       Shape Complexes in Continuous Max-Flow Hierarchical Multi-
%       Labeling Problems
%       arXiv preprint arXiv:1510.04706
    
properties
    Name
    Ct
    alpha
    D
    
    u
    
    pt
    px
    py
    lx
    ly
    div
    g

    P
    C
end

methods

    %constructor
    function h = asetsHMF2D(children,alpha,Ct)
        %add in children
        h.C = children;
        for i = 1:length(children)
            h.C{i}.P = h;
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
        for i = 1:numIts
            h.UpdateSinkFlows(cc);
            h.UpdateLabels(cc);
            h.UpdateSpatialFlows(steps,cc);
            drawnow
        end
        h.DeInitializeFullFlow();
    end

    %run pseudo-flow algorithm with set number of iterations
    function MaxPseudoFlow(h,numIts,steps,cc)
        h.InitializePseudoFlow();
        for i = 1:numIts
            h.ResetPseudoFlowCapacity();
            h.PushDownPseudoFlows();
            h.g = zeros(h.D);
            h.UpdatePseudoFlowLabels(cc,h);
            h.NormalizePseudoFlowLabels(h);
            h.PushUpCapacityReqs(h);
            h.UpdateSpatialPseudoFlows(steps,cc);
        end
        h.DeInitializePseudoFlow();
    end
    
    
    %initialize buffers for full flow
    function InitializeFullFlow(h)
        for i = 1:length(h.C)
            h.C{i}.InitializeFullFlow();
        end
        h.g = zeros(h.D, 'like', h.Ct);
        h.pt = zeros(h.D, 'like', h.Ct);
        h.u = zeros(h.D, 'like', h.Ct);
        h.px = zeros([h.D(1)-1 h.D(2)], 'like', h.Ct);
        h.py = zeros([h.D(1) h.D(2)-1], 'like', h.Ct);
        h.div = zeros(h.D, 'like', h.px);
        
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
        clear h.px;
        clear h.py;
        clear h.div;
    end
    
    %initialize buffers for pseudo flow
    function InitializePseudoFlow(h)
        for i = 1:length(h.C)
            h.C{i}.InitializePseudoFlow();
        end
        h.g = zeros(h.D, 'like', h.Ct);
        clear h.pt;
        h.px = zeros([h.D(1)-1 h.D(2)], 'like', h.Ct);
        h.py = zeros([h.D(1) h.D(2)-1], 'like', h.Ct);
        h.div = zeros(h.D, 'like', h.Ct);
        if ~isempty(h.C)
            h.u = [];
        end
        
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
    
    %deinitialize buffers for pseudo flow
    function DeInitializePseudoFlow(h)
        for i = 1:length(h.C)
            h.C{i}.DeInitializePseudoFlow();
            if i == 1
                h.u = h.C{i}.u;
            else
                h.u = h.u + h.C{i}.u;
            end
        end
        clear h.g;
        clear h.px;
        clear h.py;
        clear h.div;
    end
    
    
    %update labels (top-down)
    function UpdateLabels(h,cc)
        if ~isempty(h.P)
            h.u = h.u + cc*(h.P.pt - h.div - h.pt);
        end
        for i = 1:length(h.C)
            h.C{i}.UpdateLabels(cc);
        end
        
        if length(h.Name)>0
            PlotResults(h.Name,h.px,h.py,h.pt,h.div,h.g,h.u)
        end
    end

    %update spatial flows (top-down)
    function UpdateSpatialFlows(h,steps,cc)
        if ~isempty(h.P)

            %gradient descent on flows
            h.g = steps*( h.div + h.pt - h.P.pt - h.u/cc );
            h.px = h.px + (h.g(2:h.D(1),:)-h.g(1:h.D(1)-1,:));
            h.py = h.py + (h.g(:,2:h.D(2))-h.g(:,1:h.D(2)-1));

            %find flow mag, exemption amounts, and correction
            if numel(h.lx) > 0
                
                %find exemption amount
                a = zeros(h.D,'like',h.Ct);
                a(2:h.D(1),:)   =                  max((h.px<0).*h.px.*h.lx(2:h.D(1),:)  ,0);
                a(1,:) = 0;
                a(1:h.D(1)-1,:) = a(1:h.D(1)-1,:)+ max((h.px>0).*h.px.*h.lx(1:h.D(1)-1,:),0);
                a(:,1:h.D(2)-1) = a(:,1:h.D(2)-1)+ max((h.py>0).*h.py.*h.ly(:,1:h.D(2)-1),0);
                a(:,2:h.D(2))   = a(:,2:h.D(2))  + max((h.py<0).*h.py.*h.ly(:,2:h.D(2))  ,0);
                
                %find exemption amount
                ex =      (h.px>0 & h.lx(1:h.D(1)-1,:,:)>0).*h.lx(1:h.D(1)-1,:,:).*a(1:h.D(1)-1,:,:);
                ex = ex + (h.px<0 & h.lx(2:h.D(1),:,:)<0).*h.lx(2:h.D(1),:,:).*a(2:h.D(1),:,:);
                
                ey =      (h.py>0 & h.ly(:,1:h.D(2)-1,:)>0).*h.ly(:,1:h.D(2)-1,:).*a(:,1:h.D(2)-1,:);
                ey = ey + (h.py<0 & h.ly(:,2:h.D(2),:)<0).*h.ly(:,2:h.D(2),:).*a(:,2:h.D(2),:);
                
                %apply exemption
                h.px = h.px - ex;
                h.py = h.py - ey;
                
                %find flow mag
                h.g(1:h.D(1)-1,:) = h.px.^2;
                h.g(h.D(1),:) = 0;
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
                h.g(h.D(1),:) = 0;
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
           
        %pass on to children                 
        for i = 1:length(h.C)
            h.C{i}.UpdateSpatialFlows(steps,cc);
        end
    end
    
    %update sink flows bottom up
    function UpdateSinkFlows(h,cc)
        
        %pass call down to children first
        for i = 1:length(h.C)
            h.C{i}.UpdateSinkFlows(cc);
        end
        
        %if leaf, constrain maximum possible flow by the data cost
        if isempty(h.C)
            h.pt = min(h.Ct,h.P.pt-h.div+h.u/cc);
            
        %if source, get flow requests from children and average them
        elseif isempty(h.P)
            h.pt = 1/cc .* ones(h.D);
            for i = 1:length(h.C)
                h.pt = h.pt+h.C{i}.pt+h.C{i}.div-h.C{i}.u/cc;
            end
            h.pt = h.pt / length(h.C);
            
        %if branch, get flow requests from children as well as desired
        %sink flow and average them
        else
            h.pt = h.P.pt-h.div+h.u/cc;
            for i = 1:length(h.C)
                h.pt = h.pt+h.C{i}.pt+h.C{i}.div-h.C{i}.u/cc;
            end
            h.pt = h.pt / (1+length(h.C));
            
        end
    end

    %prepare for the push-down step
    function ResetPseudoFlowCapacity(h)
        
        %initialize capacity
        if isempty(h.C)
            h.g = h.div + h.Ct;
        elseif isempty(h.C)
            h.g = h.div;
        else
            h.g(:,:) = 0;
        end
        
        %push down call to children
        for i = 1:length(h.C)
            h.C{i}.ResetPseudoFlowCapacity();
        end
        
    end
    
    %do the push-down flow capacities step
    function PushDownPseudoFlows(h)
        for i = 1:length(h.C)
            h.C{i}.g = h.C{i}.g + h.g;
            h.C{i}.PushDownPseudoFlows();
        end
    end
    
    %update labels only at leaves
    function UpdatePseudoFlowLabels(h,cc,a)
        for i = 1:length(h.C)
            h.C{i}.UpdatePseudoFlowLabels(cc,a);
        end
        if isempty(h.C)
            h.u = h.u .* exp( - h.g / cc );
            h.g = h.u;
            a.g = a.g + h.u;
        end
    end
    
    %normalize labels only at leaves
    function NormalizePseudoFlowLabels(h,a)
        for i = 1:length(h.C)
            h.C{i}.NormalizePseudoFlowLabels(a);
        end
        if isempty(h.C)
            h.u = h.u ./ a.g;
            h.u(h.u>1) = 1; 
            h.u(h.u<0) = 0; 
            %h.g = h.u;
        end
    end
    
    %normalize labels at the leaves and push up
    function PushUpCapacityReqs(h,a)
        %normalize and clear
        if ~isempty(h.C)
            h.g = zeros(h.D);
        end
        for i = 1:length(h.C)
            h.C{i}.PushUpCapacityReqs(a);
        end
        if ~isempty(h.P) && ~isempty(h.P.P)
            h.P.g = h.P.g + h.g;
        end
    end
    
    %update spatial flows (top-down)
    function UpdateSpatialPseudoFlows(h,steps,cc)
        if ~isempty(h.P)

            %gradient descent on flows
            h.g = steps*h.g;
            h.px = h.px + h.g(2:h.D(1),:)-h.g(1:h.D(1)-1,:);
            h.py = h.py + h.g(:,2:h.D(2))-h.g(:,1:h.D(2)-1);

            %find flow mag, exemption amounts, and correction
            if numel(h.lx) > 0
                
                %find exemption amount
                a(1:h.D(1)-1,:) =                  max((h.px>0).*h.px.*h.lx(1:h.D(1)-1,:),0);
                a(h.D(1),:) = 0;
                a(2:h.D(1),:)   = a(2:h.D(1),:)  + max((h.px<0).*h.px.*h.lx(2:h.D(1),:)  ,0);
                a(:,1:h.D(2)-1) = a(:,1:h.D(2)-1)+ max((h.py>0).*h.py.*h.ly(:,1:h.D(2)-1),0);
                a(:,2:h.D(2))   = a(:,2:h.D(2))  + max((h.py<0).*h.py.*h.ly(:,2:h.D(2))  ,0);
                
                %find exemption amount
                ex =      (h.px>0 & h.lx(1:h.D(1)-1,:,:)>0).*h.lx(1:h.D(1)-1,:,:).*a(1:h.D(1)-1,:,:);
                ex = ex + (h.px<0 & h.lx(2:h.D(1),:,:)<0).*h.lx(2:h.D(1),:,:).*a(2:h.D(1),:,:);
                
                ey =      (h.py>0 & h.ly(:,1:h.D(2)-1,:)>0).*h.ly(:,1:h.D(2)-1,:).*a(:,1:h.D(2)-1,:);
                ey = ey + (h.py<0 & h.ly(:,2:h.D(2),:)<0).*h.ly(:,2:h.D(2),:).*a(:,2:h.D(2),:);
                
                %apply exemption
                h.px = h.px - ex;
                h.py = h.py - ey;
                
                %find flow mag
                h.g(1:h.D(1)-1,:) = h.px.^2;
                h.g(h.D(1),:) = 0;
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
                h.g(h.D(1),:) = 0;
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
           
        %pass on to children                 
        for i = 1:length(h.C)
            h.C{i}.UpdateSpatialPseudoFlows(steps,cc);
        end
    end
    
end
    
end