classdef asetsHMF2D < handle
%   John SH Baxter, Robarts Research Institute, 2015
%
%   Full implementation with of [1] in 2D and HMF implementation
%   of [2]
%
%   [1] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Continuous Max-Flow Approach to General
%       Hierarchical Multi-Labelling Problems
%       arXiv preprint arXiv:1404.0336
%
%   [2] Baxter, JSH.; Rajchl, M.; Yuan, J.; Peters, TM.
%       A Proximal Bregman Projection Approach to Continuous
%       Max-Flow Problems Using Entropic Distances
%       arXiv preprint arXiv:1501.07844
    
properties
    Ct
    alpha
    D
    
    u
    
    pt
    px
    py
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
            h.Ct = [];
        end
        
        %create buffers
        h.alpha = alpha;
        h.u = zeros(h.D);
        
    end

    %run full-flow algorithm with set number of iterations
    function MaxFullFlow(h,numIts,steps,cc)
        h.InitializeFullFlow();
        for i = 1:numIts
            h.UpdateSpatialFlows(steps,cc);
            h.UpdateSinkFlows(cc);
            h.UpdateLabels(cc);
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
        h.g = zeros(h.D);
        h.pt = zeros(h.D);
        h.px = zeros([h.D(1)-1 h.D(2)]);
        h.py = zeros([h.D(1) h.D(2)-1]);
        h.div = zeros(h.D);
        if ~isempty(h.C)
            h.u = zeros(h.D);
            for i = 1:length(h.C)
                h.u = h.u + h.C{i}.u;
            end
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
        h.g = zeros(h.D);
        clear h.pt;
        h.px = zeros([h.D(1)-1 h.D(2)]);
        h.py = zeros([h.D(1) h.D(2)-1]);
        h.div = zeros(h.D);
        if ~isempty(h.C)
            h.u = [];
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
    end

    %update spatial flows (top-down)
    function UpdateSpatialFlows(h,steps,cc)
        if ~isempty(h.P)

            %gradient descent on flows
            h.g = steps*( h.div + h.pt - h.P.pt - h.u/cc );
            h.px = h.px + h.g(2:end,:)-h.g(1:end-1,:);
            h.py = h.py + h.g(:,2:end)-h.g(:,1:end-1);

            %find flow mag
            h.g = zeros(size(h.div));
            h.g(1:end-1,:) = h.g(1:end-1,:) + h.px.^2;
            h.g(2:end,:) = h.g(2:end,:) + h.px.^2;
            h.g(:,1:end-1) = h.g(:,1:end-1) + h.py.^2;
            h.g(:,2:end) = h.g(:,2:end) + h.py.^2;
            h.g = h.g .^ 0.5;

            %correct for flow mag
            mask = (h.g <= h.alpha);
            if numel(h.alpha) == 1
                h.g(~mask) = h.alpha ./ h.g(~mask);
            else
                h.g(~mask) = h.alpha(~mask) ./ h.g(~mask);
            end
            h.g(mask) = 1;
            h.px = 0.5 * h.px .* (h.g(2:end,:)+h.g(1:end-1,:));
            h.py = 0.5 * h.py .* (h.g(:,2:end)+h.g(:,1:end-1));

            %calculate divergence
            h.div = zeros(size(h.div));
            h.div(1:end-1,:) = h.div(1:end-1,:) + h.px;
            h.div(2:end,:) = h.div(2:end,:) - h.px;
            h.div(:,1:end-1) = h.div(:,1:end-1) + h.py;
            h.div(:,2:end) = h.div(:,2:end) - h.py;
            
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
            h.pt = 1/cc .* ones(size(h.pt));
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
            h.px = h.px - h.g(2:end,:)+h.g(1:end-1,:);
            h.py = h.py - h.g(:,2:end)+h.g(:,1:end-1);

            %find flow mag
            h.g = zeros(size(h.div));
            h.g(1:end-1,:) = h.g(1:end-1,:) + h.px.^2;
            h.g(2:end,:) = h.g(2:end,:) + h.px.^2;
            h.g(:,1:end-1) = h.g(:,1:end-1) + h.py.^2;
            h.g(:,2:end) = h.g(:,2:end) + h.py.^2;
            h.g = h.g .^ 0.5;

            %correct for flow mag
            mask = (h.g <= h.alpha);
            if numel(h.alpha) == 1
                h.g(~mask) = h.alpha ./ h.g(~mask);
            else
                h.g(~mask) = h.alpha(~mask) ./ h.g(~mask);
            end
            h.g(mask) = 1;
            h.px = 0.5 * h.px .* (h.g(2:end,:)+h.g(1:end-1,:));
            h.py = 0.5 * h.py .* (h.g(:,2:end)+h.g(:,1:end-1));

            %calculate divergence
            h.div = zeros(size(h.div));
            h.div(1:end-1,:) = h.div(1:end-1,:) + h.px;
            h.div(2:end,:) = h.div(2:end,:) - h.px;
            h.div(:,1:end-1) = h.div(:,1:end-1) + h.py;
            h.div(:,2:end) = h.div(:,2:end) - h.py;
            
        end
           
        %pass on to children                 
        for i = 1:length(h.C)
            h.C{i}.UpdateSpatialPseudoFlows(steps,cc);
        end
    end
    
end
    
end

