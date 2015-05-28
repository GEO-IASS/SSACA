function seg = dpSearchW(S,W, para, seg0, segT, mustLink, cannotLink)
% Search for the optimal segementation via dynamic programming.
%
% Input
%   S       -  frame similarity matrix, n x n
%   para    -  parameter of segmentation
%     lc    -  local constraint (Sakoe and Chiba)
%              0 : not used
%              1 : used
%     wFs   -  frame weights, {[]}
%   seg0    -  inital segmentation
%
% Output
%   seg     -  segmentation
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 12-29-2008
%   modify  -  Rodrigo Araujo (sineco@gmail.com), 28-12-2013

n = size(S, 1);

% local constarint
lc = ps(para, 'lc', 0);

% frame weights
wFs = ps(para, 'wFs', []);
if isempty(wFs)
    wFs = ones(1, n);
end

% pairwise DTAK
T = dtaksFord(S, seg0, lc, wFs);

% Visualization only
% k = 2;
% X = eigk(T, k);
% Y = X';
% figure;
% plot(Y(1,:), Y(2,:), 'o');
% for i=1:size(Y,2)
%     text(Y(1,i), Y(2,i), num2str(i),'FontSize',18)
% end

% Form K - Type 1
sizeT = size(T,1);
G = eye(sizeT);
%Ws = constructConstraintMatrixSegment(seg0.s, W, segT.s); %Re implement this function later - It's working fine but not optimized. 
% Remap constraints 
[headTail, remapedSegConstrains, mustLink, cannotLink, Ws]=remapConstraints(seg0, segT.s, mustLink, cannotLink);
K = T + Ws;                     

%Form K - Type 2
%Replace the the respective weights (1 or -1) for the known segments
% T(find(Ws~=0)) = Ws(find(Ws~=0)); %its working
% K = T;                            %its working

%Form K - Type 3
% Normalize weight
% constraints = [mustLink; cannotLink];
% C = size(constraints, 1);
% k = para.k;
% N = size(T,1);
% T(find(Ws>0)) = N/(k*C);
% T(find(Ws<0)) = -1*(N/(k*C));
% K = T;

% Test positive definitiness
positiveDefinite = all(eig(K) > 0);
positiveDefinite = 1; %!!!!!!Nooooot doing shiiiiift!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Without doing the convergence there is no guarantee that it will converge
if ~positiveDefinite
    %Diagonal-shift K by adding ?I to guarantee positive definiteness of K
%      sigma = abs(min(eig(K))); %+ 0.00000000000001 + 0.000000000000001;
    sigma = 0.24;
%       sigma = -0.0000000000000001;
%     sigma = -10; % 
    K = K + sigma*eye(size(K,1));
end

pd = all(eig(K) > 0);

%Form K frame level
Kf = S;
% Remap constraints 
% [headTail, remapedSegConstrains, mustLink, cannotLink]=remapConstraints(seg0, segT.s, mustLink, cannotLink);
% forward 
% [sOpt, cOpt, objOpt, distsOpt] = sacaFordSlow(Kf, para, seg0, K, lc, wFs, headTail, remapedSegConstrains);
[sOpt, cOpt, objOpt] = sacaFord(Kf, para, seg0, K, lc, wFs,headTail, remapedSegConstrains);

% backward
seg = acaBack(para.k, sOpt, cOpt);
% seg = acaBackSlow(para.k, sOpt, cOpt);

% objective value
seg.obj = objOpt(end);

% 
%headsSeg = seg.s;
%tailsSeg = headsSeg - 1;
%seg.dists = distsOpt(tailsSeg(2:end), :);
% seg.dists = distsOpt(end);
