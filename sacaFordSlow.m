function [sOpt, cOpt, objOpt, distsOpt] = sacaFordSlow(K, para, seg, T, lc, wFs, headTail, remapedSegConstrains)
% Forward step in ACA.
%
% Input
%   K       -  kernel matrix, n x n
%   seg     -  inital segmentation
%   para    -  segmentation parameter
%   T       -  pairwise DTAK, m x m
%   lc      -  local constraint (Sakoe and Chiba)
%              0 : not used
%              1 : used
%   wFs     -  frame weights, 1 x n
%
% Output
%   sOpt    -  optimum starting position, n x 1
%   cOpt    -  optimum label, n x 1
%   objOpt  -  optimum objective, n x 1
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 12-29-2008
%   modify  -  Feng Zhou (zhfe99@gmail.com), 01-13-2010
%   modify  -  Rodrigo Araujo (sineco@gmail.com), 08-10-2014

if lc == 0
    [sOpt, cOpt, objOpt, distsOpt] = acaDP(K, para, seg, T, wFs, headTail, remapedSegConstrains);
elseif lc == 1
    [sOpt, cOpt, objOpt, distsOpt] = acaDPLc(K, para, seg, T, wFs);
else
    error('unknown local constraint');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sOpt, cOpt, objOpt, distsOpt] = acaDP(K, para, seg, T, wFs,headTail, remapedSegConstrains)
% Move forward to construct the path matrix without local constraint.
%
% Input
%   K         -  kernel matrix, n x n
%   seg       -  inital segmentation
%   para      -  segmentation parameter
%   T         -  pairwise DTAK, m x m
%   wFs       -  frame weights, 1 x n
%   remapedSegConstrains - the list of segments indices that has some
%   constraint attached 
%   headTail  - segment indices [from to]
%
% Output
%   sOpt    -  optimum starting position, 1 x n
%   cOpt    -  optimum label, 1 x n
%   objOpt  -  optimum objective, 1 x n
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 12-29-2008
%   modify  -  Feng Zhou (zhfe99@gmail.com), 01-13-2010
%   modify  -  Rodrigo Araujo (sineco@gmail.com), 08-10-2014, added
%   semi-supervised support



nMa = para.nMa;
s = seg.s; G = seg.G;

n = s(end) - 1; [k, m] = size(G);
l = G2L(G);
Q = zeros(nMa, nMa, n);
[sOpt, cOpt, objOpt] = zeross(n, 1);
% New output 
distsOpt = zeross(n,para.k);
ws = zeros(1, m);
for i = 1 : m
    ws(i) = sum(wFs(s(i) : s(i + 1) - 1));
end

% 3rd component in the kernel expansion of distance
t3 = diag((G * G') \ (G * T * G') / (G * G'));
head = 0; %new
tail = 0; %new
count = 0; %new
for v = 1 : n
    % initial value
    sOpt(v) = 0; cOpt(v) = 0; objOpt(v) = -1; distsOpt(v,:) = -1*ones(1, para.k);
    
    % if v is inside a known sequence skip the processing and go straight
    % to the tail of the sequence
    inTheWindow = 0;
    isTail = 0;
    for ind = 1:size(headTail,1)
        if v >= headTail(ind,1) && v < headTail(ind,2)
            inTheWindow = 1;
            
        end
        if v == headTail(ind,2)
            isTail = 1;
            head = headTail(ind,1);
            tail = headTail(ind,2);
        end
    end
    
    if inTheWindow
        continue;
    end
    
    % Save processing
    wv = 0; t1a = 0; 
    % After skipping the known part, calculate directly the DTAK for X[i,v]
    if isTail %if gets in here it is necessarily in a boost mode
        % current head
        count = count + 1;
        i = head;
        wv = wv + wFs(i);
        % 1st component in the kernel expansion of distance
        t1a = t1a + wFs(i) * K(i, i);
        t1 = t1a / wv;
        % 2nd component in the kernel expansion of distance
        t2 = zeros(k, 1);
        for j = 1 : m
            % Compute t = (X[i,v], Yj);
            lc = 0; % 0 or 1
            ii = [head:v];
            jj = s(j) : s(j + 1) - 1;
            [P, KC] = zeross(size(ii,2), size(jj,2));
            % Output
            %   v       -  dtak value
            %   P       -  path matrix, n1 x n2
            %   KC      -  cummulative similarity matrix, n1 x n2
%             [q, P, KC] = dtakFordSlow(K(ii, jj), lc, wFs(ii), wFs(jj));
            q = T(remapedSegConstrains(count), j);
            % Update U(nv,vbar, sdot);
%             Q = updateQ(KC, Q, ii, jj, nMa); (Not necessary for the result)
            
            t2(l(j)) = t2(l(j)) + q; %T2 is like a distance too
        end
        t2 = (G * G') \ t2 * 2;
        
        % distance
        dists = t1 - t2 + t3;
        [distMi, cMi] = min(dists);
        
        i = head;
        
        dist = distMi;
        
        sOpt(i:v) = i;
        cOpt(i:v) = cMi;
        objOpt(i:v) = dist;
        distsOpt(i:v,:) = repmat(dists', (v-i)+1, 1);
        
    else
        vBar = mod(v, nMa) + 1;
        v1Bar = mod(v - 1, nMa) + 1;
        
        wv = 0; t1a = 0;
        for nv = 1 : min(nMa, v)
            
            % current head
            i = v - nv + 1;
            wv = wv + wFs(i);
            
            % 1st component in the kernel expansion of distance
            t1a = t1a + wFs(i) * K(i, i);
            t1 = t1a / wv;
            
            % 2nd component in the kernel expansion of distance
            t2 = zeros(k, 1);
            for j = 1 : m
                for sDot = s(j) : s(j + 1) - 1 %rsaa: I know every subsequence that is in the must-link and cannot link
                    kk = K(v, sDot);
                                        
                    if nv == 1 && sDot == s(j)
                        Q(nv, vBar, sDot) = (wFs(v) + wFs(sDot)) * kk;
                    elseif nv == 1
                        Q(nv, vBar, sDot) = Q(nv, vBar, sDot - 1) + wFs(sDot) * kk;
                    elseif sDot == s(j)
                        Q(nv, vBar, sDot) = Q(nv - 1, v1Bar, sDot) + wFs(v) * kk;
                    else
                        a = Q(nv - 1, v1Bar, sDot - 1) + (wFs(v) + wFs(sDot)) * kk;
                        b = Q(nv, vBar, sDot - 1) + wFs(sDot) * kk;
                        c = Q(nv - 1, v1Bar, sDot) + wFs(v) * kk;
                        Q(nv, vBar, sDot) = max([a, b, c]);
                    end
                end
                
                t2(l(j)) = t2(l(j)) + Q(nv, vBar, s(j + 1) - 1) / (wv + ws(j)); %T2 is like a distance too
            end
            t2 = (G * G') \ t2 * 2;
            
            % distance
            dists = t1 - t2 + t3;
            
            [distMi, cMi] = min(dists);
            
            i = v - nv + 1;
            if i == 1 || sOpt(i - 1) > 0 %~= 0 since if it is not processed it will be zero
                
                if (i == 1) || (i - 1 == tail) %if stating a new segment after a known segment
                    dist = distMi;
                else
                    dist = objOpt(i - 1) + distMi;
                end
                
                if dist < objOpt(v) || objOpt(v) < 0% rsaa: distsOpt. we would need to have an if like that for the distance to the centroid either. I might as well just use the objective function.
                    sOpt(v) = i;
                    cOpt(v) = cMi;
                    objOpt(v) = dist;
                    % This new output represents the distances of each
                    % point to the centres of the clusters. It is
                    % cumulative, just like the objective function is, in the
                    % forward step. In the backward step, we will retrieve
                    % the minimized values of the distances, just like we
                    % do with the objective functions.
                    distsOpt(v,:) = dists;
                end
                
                % stop processing when it analyzes an interval that overlaps a
                % known segment
                if i - 1 == tail %the tail found on the last known segment
                    break;
                end
            end
            
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sOpt, cOpt, objOpt, distsOpt] = acaDPLc(K, para, seg, T, wFs)
% Move forward to construct the path matrix without local constraint.
%
% Input
%   K       -  kernel matrix, n x n
%   seg     -  inital segmentation
%   para    -  segmentation parameter
%   T       -  pairwise DTAK, m x m
%   wFs     -  frame weights, 1 x n
%
% Output
%   sOpt    -  optimum starting position, 1 x n
%   cOpt    -  optimum label, 1 x n
%   objOpt  -  optimum objective, 1 x n

nMa = para.nMa;
s = seg.s; G = seg.G;

n = s(end) - 1; [k, m] = size(G);
l = G2L(G);
[Q, P] = zeross(nMa, nMa, n);
[sOpt, cOpt, objOpt] = zeross(n, 1);
% New output 
distsOpt = zeross(n,para.k);

ws = zeros(1, m);
for i = 1 : m
    ws(i) = sum(wFs(s(i) : s(i + 1) - 1));
end

% 3rd component in the kernel expansion of distance
t3 = diag((G * G') \ (G * T * G') / (G * G'));

for v = 1 : n
    % initial value
    sOpt(v) = 0; cOpt(v) = 0; objOpt(v) = -1; distsOpt(v,:) = -1*ones(1, para.k);
    vBar = mod(v, nMa) + 1;
    v1Bar = mod(v - 1, nMa) + 1;
    v2Bar = mod(v - 2, nMa) + 1;
    
    wv = 0; t1a = 0;
    for nv = 1 : min(nMa, v)
        % current head
        i = v - nv + 1;
        wv = wv + wFs(i);
        
        % 1st component in the kernel expansion of distance
        t1a = t1a + wFs(i) * K(i, i);
        t1 = t1a / wv;
        
        % 2nd component in the kernel expansion of distance
        t2 = zeros(k, 1);
        for j = 1 : m
            for sDot = s(j) : s(j + 1) - 1
                kk = K(v, sDot);
                
                Q(nv, vBar, sDot) = -1;
                P(nv, vBar, sDot) = -1;
                
                if nv == 1 && sDot == s(j)
                    Q(nv, vBar, sDot) = (wFs(v) + wFs(sDot)) * kk;
                    P(nv, vBar, sDot) = 0;
                    
                elseif nv == 1
                    
                elseif sDot == s(j)
                    
                elseif nv == 2 || sDot == s(j) + 1
                    if P(nv - 1, v1Bar, sDot - 1) >= 0
                        Q(nv, vBar, sDot) = Q(nv - 1, v1Bar, sDot - 1) + (wFs(v) + wFs(sDot)) * kk;
                        P(nv, vBar, sDot) = 3;
                    end
                    
                else
                    if P(nv - 1, v1Bar, sDot - 1) >= 0
                        Q(nv, vBar, sDot) = Q(nv - 1, v1Bar, sDot - 1) + (wFs(v) + wFs(sDot)) * kk;
                        P(nv, vBar, sDot) = 3;
                    end
                    
                    tmp = Q(nv - 1, v1Bar, sDot - 2) + (wFs(v) + wFs(sDot - 1)) * K(v, sDot - 1) + wFs(sDot) * kk;
                    if P(nv - 1, v1Bar, sDot - 2) >= 0 && tmp > Q(nv, vBar, sDot)
                        Q(nv, vBar, sDot) = tmp;
                        P(nv, vBar, sDot) = 4;
                    end
                    
                    tmp = Q(nv - 2, v2Bar, sDot - 1) + (wFs(v - 1) + wFs(sDot)) * K(v - 1, sDot) + wFs(v) * kk;
                    if P(nv - 2, v2Bar, sDot - 1) >= 0 && tmp > Q(nv, vBar, sDot)
                        Q(nv, vBar, sDot) = tmp;
                        P(nv, vBar, sDot) = 5;
                    end
                end
            end
            t2(l(j)) = t2(l(j)) + Q(nv, vBar, s(j + 1) - 1) / (wv + ws(j));
        end
        t2 = (G * G') \ t2 * 2;
        
        % distance
        dists = t1 - t2 + t3;
        
        [distMi, cMi] = min(dists);
        
        i = v - nv + 1;
        if i == 1 || sOpt(i - 1) > 0
            
            if i == 1
                dist = distMi;
            else
                dist = objOpt(i - 1) + distMi;
            end
            
            if dist < objOpt(v) || objOpt(v) < 0
                sOpt(v) = i;
                cOpt(v) = cMi;
                objOpt(v) = dist;
                distsOpt(v,:) = dists;
            end
        end
    end
end
