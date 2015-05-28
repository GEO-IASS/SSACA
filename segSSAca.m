function [seg, segs] = segSSAca(S, W, para, seg0,segTruth, mustLink, cannotLink)
% Semi Supervised Aligned Cluster Analysis (SSACA).
% See the following paper for the details:
%   Aligned Cluster Analysis for Temporal Segmentation of Human Motion, FG 2008
%
% Input
%   S          -  
%   K          -  frame similarity matrix, n x n
%   para       -  parameter of segmentation
%   seg0       -  inital segmentation
%   segTruth   -  ground truth segmentation
%   must-link  -  pairs of must-link
%   cannot-link-  pairs of cannot-link
%
% Output
%   seg     -  segmentation result
%   segs    -  segmentation result during the procedure, 1 x nIter (cell)
%
% History
%   created  -  Rodrigo Araujo (sineco@gmail.com), 24-12-2013

% maximum number of iterations
nIterMa = 20;

segs = cell(1, nIterMa);
tim = 0;
for nIter = 1 : nIterMa
    % search
    segs{nIter} = dpSearchW(S,W, para, seg0, segTruth, mustLink, cannotLink);
        

    % stop condition
    if cluEmp(segs{nIter}.G)
        prom('b', 'segAca stops due to an empty cluster\n');
        %segs{nIter}.obj = inf; %rsaa I commented that because I want to
        %analyse the behaviour of the objective function even if it
        %returned a non specified number of clusters.
        break;
    elseif isequal(segs{nIter}.G, seg0.G) && isequal(segs{nIter}.s, seg0.s)
        break;
    end

    seg0 = segs{nIter};
end
segs(nIter + 1 : end) = [];

segs{nIter}.tim = tim / nIter;
seg = segs{nIter};
prom('b', 'dpSearch : %.2f seconds, %d iterations\n', seg.tim, nIter);
