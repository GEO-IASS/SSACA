function [seg, segs, objs, accs, iOpt, evalParam] = segAlg(alg, X, K, para, seg0s, segT, mustLink, cannotLink, W)
% Temporal segmentation via specific algorithm.
%
% Input
%   alg     -  algorithm name, 'aca' | 'haca' | 'gmm' | 'slds'
%   X       -  sequence, dim x n
%   K       -  kernel matrix, n x n
%   para    -  segmentation parameter
%   seg0s   -  initial segmentation, 1 x nIni (cell)
%   segT    -  ground-truth segmentation (can be empty)
%
% Output
%   seg     -  segmentation with minimum cost: seg = segs{iOpt}, where iOpt = argmin_i costs(i)
%   segs    -  all segmentations, 1 x nIni (cell)
%   objs    -  ojbective, 1 x nIni
%   accs    -  accuracy (if segT is specified), 1 x nIni
%   iOpt    -  the index of optimal initial segmentation
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-05-2009
%   modify  -  Rodrigo Araujo (sineco@gmail.com), 01-03-2014 -  added ssaca, sssc,
%   e2cp, kms

nIni = length(seg0s);
segs = cell(1, nIni);
[objs, accs] = zeross(1, nIni);
for i = 1 : nIni
    prom('m', 'new segmentation (%s): %d time\n', alg, i);

    % initialization
    seg0 = seg0s; %seg0s{i}

    % algorithm
    if strcmp(alg, 'aca')
        seg = segAca(K, para, seg0);
        objs(i) = seg.obj;

    elseif strcmp(alg, 'haca')
        seg = segHaca(K, para, seg0);
        objs(i) = seg(end).obj;
    elseif strcmp(alg, 'ssaca')
        seg = segSSAca(K, W, para, seg0, segT, mustLink, cannotLink);
%         seg = segSSAca(K, para, seg0);
        objs(i) = seg(end).obj;

    elseif strcmp(alg, 'gmm')
        seg = segGmm(K, para);
        
    elseif strcmp(alg, 'sssc')
        seg = segSSsc(K, para, segT, mustLink, cannotLink);    
     
    elseif strcmp(alg, 'e2cp')
        seg = segE2cp(K, [], para, seg0, segT, mustLink, cannotLink);
%         [seg, evalParam] = segE2cpTemporal(K, para, seg0, segT, mustLink, cannotLink);
        evalParam = [];                    
    elseif strcmp(alg, 'sl')
        seg = segSL(K, [], para, seg0, segT, mustLink, cannotLink);
          evalParam = [];  
    elseif strcmp(alg, 'kms')
        seg = segKmsTemporal(K, X, para, seg0, segT, mustLink, cannotLink);
                      
    elseif strcmp(alg, 'hmm')
        seg = emSto(X, para, seg0, 'sto', 'hmm');

    elseif strcmp(alg, 'lds')
        seg = emSto(X, para, seg0, 'sto', 'lds');
    
    elseif strcmp(alg, 'tdck')
        seg = segTdck(X, para);
    else
        error('unknown function');
    end

    % accuracy
    if ~isempty(segT)
        if strcmp(alg, 'haca')
            seg(end) = matchSeg(seg(end), segT(end));
            accs(i) = seg(end).acc;
        else
            seg = matchSeg(seg, segT);
            accs(i) = seg.acc;
        end
    end

    % record
    segs{i} = seg;
end

% choose the result with minimum objective value
if strcmp(alg, 'aca') || strcmp(alg, 'haca')
    [obj, iOpt] = min(objs);
else
    iOpt = 1;
end
seg = segs{iOpt};

