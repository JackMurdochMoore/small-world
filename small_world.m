% Generalisation of Watts-Strogatz (WS) small world model described in：
% 
% Haiying Wang, Jack Murdoch Moore, Michael Small, Jun Wang, Huijie Yang,
% Changgui Gu, Epidemic dynamics on higher-dimensional small world
% networks, Applied Mathematics and Computation, Volume 421, 2022, 126911.
% https://doi.org/10.1016/j.amc.2021.126911
% https://www.sciencedirect.com/science/article/pii/S0096300321009942
%
% Generate adjacency matrix of a version of WS small world
% network with any positive integer dimension.
%
% The probability p_j that
% a node j is chosen to be the target of a rewired edge can depend on
% distance d as    
% p_j ~ 1/(kappa + omega*d^-sigma). -- (1)
% 
% Required inputs:
%   N - approximate network size (positive integer)
%   k - mean degree (even positive integer)
%   D - dimension (positive integer)
%   p - rewiring rate (0 <= p <= 1)
% 
% Optional inputs:
%   kappa, omega, sigma -   parameters controlling dependence on
%                           (non-network) distance (see Eq. (1) above).
%   rewireFlag - whether to rewire or add links:
%               == 1 =>     rewire links (mean degree independent of
%                           rewiring rate p) 
%               == 0 =>     add links (mean degree increases with rewiring
%                           rate p) 
%   lowerAndUpperQuantile - Control (non-network) distances spanned by
%                           links resulting from rewiring or adding links.
%                           When lowerAndUpperQuantile == [a, b], no links
%                           are rewired such that the (non-network)
%                           distance between target and recipient is in
%                           lower fraction a or upper fraction (1 - b).
%
% Output:
%   A - adjacency matrix
% 
% Example use:
% N = 400; k = 6; D = 2; p = 0.05; A = small_world(N, k, D, p); G = graph(A); figure;
% N = 400; k = 6; D = 2; p = 0.05; sigma = 3; kappa = 1; omega = 0; rewireFlag = 1; lowerAndUpperQuantile = [-eps, 1 + eps]; A = small_world(N, k, D, p, sigma, kappa, omega, rewireFlag, lowerAndUpperQuantile); G = graph(A); figure; plot(G, 'Layout', 'Force3');
% N = 2000; k = 6; D = 3; p = 0; sigma = 0; kappa = 1; omega = 0; rewireFlag = 1; lowerAndUpperQuantile = [-eps, 1 + eps]; A = small_world(N, k, D, p, sigma, kappa, omega, rewireFlag, lowerAndUpperQuantile); G = graph(A); figure; plot(G, 'Layout', 'Force3');
%
% Notes:
% 1. When p = 0 the graph is regular (each node has the same degree).
% 2. When either omega = 0 or sigma = 0 (and kappa > 0) there is no
%    dependence on distance. 
% 
% This code is provided without warranty or guarantee.
% 
% If you use this code in a publication, project, etc., then please cite：
% 
% Haiying Wang, Jack Murdoch Moore, Jun Wang, and Michael Small, "The
% distinct roles of initial transmission and retransmission in the
% persistence of knowledge in complex networks," Applied Mathematics and
% Computation *vol* (*year*) *page*.
% 
% Jack Murdoch Moore, 2021-12-30
% 
function A = small_world(N, k, D, p, varargin)
L = round(N^(1/D));
N = L^D;

numOptArgs = numel(varargin);

if (numOptArgs < 1)
    sigma = 0;
else
    sigma = varargin{1};
end
if (numOptArgs < 2)
    kappa = 1;
else
    kappa = varargin{2};
end
if (numOptArgs < 3)
    omega = 0;
else
    omega = varargin{3};
end
if (numOptArgs < 4)
    rewireFlag = 1;
else
    rewireFlag = varargin{4};
end
if (numOptArgs < 5)
    lowerAndUpperQuantile = [-eps, 1 + eps];
else
    lowerAndUpperQuantile = varargin{5};
end

assert((kappa >= 0) && (sigma >= 0), 'Please choose non-negative parameters kappa and sigma.');
assert((kappa > 0) || (sigma > 0), 'Please choose either (or both) kappa or sigma positive.');
assert(mod(k, 2) == 0, 'Please choose an even positive integer degree k.');

inCell = repmat({1:L}, [1, D]);% D-dimensional grid with side lengths L

% 1. First compute adjacency matrix for a regular network. All nodes within
% distance r of each other are connected, where r is the smallest number
% for which the network degree is greater or equal to k.

positionCell = cell(1, D);%Store position in each dimension
[positionCell{:}] = ndgrid(inCell{:});
for ii = 1:D
    out = positionCell{ii};
    positionCell{ii} = out(:);
end
distFun0 = @(distX) min(distX, L - distX);
distFun = @(X) distFun0(squareform( pdist(X) ));

distList = cellfun(distFun, positionCell, 'UniformOutput', false);
distList = reshape(distList, [1, 1, D]);
distList = cell2mat(distList);

% Choose definition of distance:
% q = 1;%L^1/city block/Manhattan distance
q = 2;%L^2/Euclidean distance
% q = Inf;%Supremum distance

%Distances between nodes:
distMat = vecnorm(distList, q, 3);

%Distances from first node:
distances = distMat(1, :);

%Determine radius within which there are at least k nodes:
deg = @(r) sum((distances <= r) & (distances > 0));
rUpper = 0; degUpper = 0;
while (degUpper <= k)
    rUpper = rUpper + 1;
    degUpper = deg(rUpper);
end

%Determine smallest radius r (to within sigma_r) within which there are at least k nodes:
sigma_r = 0.01;
rr = 0:sigma_r:rUpper;
num_r = numel(rr);
kk = NaN(1, num_r);
for ii_r = 1:num_r
    r = rr(ii_r);
    kk(ii_r) = deg(r);
end
ii_k = find(kk >= k, 1);
r = rr(ii_k);
rL = rr(ii_k - 1);

%Link all nodes within distance r:
A = ( ( distMat <= r ) & ( distMat > 0 ) );

% 2. Next, remove links between most distant neighbours to reach a regular
% network with degree k. 

m = 0.5*sum(A(:))/N;%Total number of links divided by total number of nodes before random deletion

nodeOrderEachDim = cell2mat(positionCell);

numToDelete = 2*m - k;%Number to delete per node

mostDistNeighbours = find((distances <= r) & (distances > rL));

neighbToDelete = datasample(mostDistNeighbours, numToDelete, 'Replace', false);

subCellNeighbToDelete = cell(1, D);
[subCellNeighbToDelete{:}] = ind2sub(L*ones(1, D), neighbToDelete');
subNeighbToDelete = cell2mat(subCellNeighbToDelete);
sigmaSubNeighbToDelete = subNeighbToDelete - 1;

A0 = A;%Adjacency matrix before deletions

% Successively remove links between the most distant neighbours (in the
% same way for each node) until all the degree of the (still regular)
% network decreases to k.
for kk = 1:numToDelete
    for ii = 1:N
        subCellCurrentNeighb = cell(1, D);
        [subCellCurrentNeighb{:}] = ind2sub(L*ones(1, D), ii);
        subCurrentNeighb = cell2mat(subCellCurrentNeighb);
        for hh = 1:D
            subNeighbToDelete(:, hh) = mod(subCurrentNeighb(hh) + sigmaSubNeighbToDelete(:, hh) - 1, L) + 1;
        end
        subCellNeighbToDelete = mat2cell(subNeighbToDelete, numToDelete, ones(1, D));
        neighbToDelete = sub2ind(L*ones(1, D), subCellNeighbToDelete{:});
        jj = neighbToDelete(kk);
        A(ii, jj) = 0; A(jj, ii) = 0;
    end
    if (sum(A(:)) <= k*N); break; end%If desired degree is reached then break the loop to stop deleting links.
end

% 3. Finally, randomly rewire some edges. After rewiring, each node will
% have degree at least k/2.

lowerAndUpperQuantile;
lowQuant = lowerAndUpperQuantile(1); uppQuant = lowerAndUpperQuantile(2);
if lowQuant <= 0
    lowDist = 0;
else
    lowDist = quantile(distances, lowQuant);
end
if uppQuant >= 1
    uppDist = Inf;
else
    uppDist = quantile(distances, uppQuant);
end

A0 = A;%Adjacency matrix before rewiring
edgeList0 = adj_to_edge(A0);%Edge list before rewiring
numEdges = size(edgeList0, 1);
for iiEdge = 1:numEdges
    if (rand() < p)%If rand() < p then rewire target of edge currently considered
        edge = edgeList0(iiEdge, :);
        ii = edge(1);
        jj = edge(2);
        a = edge(3);
        %Should ii or jj be considered the target?
        nodeOrderEachDim_ii = nodeOrderEachDim(ii, :);
        nodeOrderEachDim_jj = nodeOrderEachDim(jj, :);
        diffPosDim = find(nodeOrderEachDim_ii - nodeOrderEachDim_jj, 1, 'last');
        pos_ii = nodeOrderEachDim_ii(diffPosDim);
        pos_jj = nodeOrderEachDim_jj(diffPosDim);
        d_jj_ii = (pos_jj - pos_ii);
        d_jj_ii = min(d_jj_ii, L + d_jj_ii);
        if (d_jj_ii > L/2)%If necessary, swap ii and jj so that jj is the target
            kk = jj;
            jj = ii;
            ii = kk;
        end
        if (sum(A(jj, :)) <= ii)
            ':I';
        end
        candidateNewTargets = find((A(ii, :) == 0) & ((1:N) ~= ii));
        distCandidateNewTargets = distMat(ii, candidateNewTargets);
        candidateNewTargets = candidateNewTargets((distCandidateNewTargets >= lowDist) & ((distCandidateNewTargets <= uppDist)));
        if rewireFlag
            A(ii, jj) = 0; A(jj, ii) = 0;%Remove old link
        end
%         candidateNewTargets = [candidateNewTargets, jj];%Have this line commented (uncommented) to forbid (allow) a rewiring step which comprises adding the link which was just removed
        dd = distMat(ii, candidateNewTargets);%Distances to candidate targets
        ww = 1./(kappa + omega*(dd.^sigma));%Weights for probabilities of choosing new neighbour
        jj = datasample(candidateNewTargets, 1, 'Weights', ww);%Choose new target
        A(ii, jj) = a; A(jj, ii) = a;%Add new link to new target
    end
end

assert(k == sum(A(:))/size(A, 1), 'There is a problem. The final mean degree k does not match the intended mean degree. Perhaps N is too small or D is too large to incorporate this value of k.');

end

%Find (randomly ordered) edge list from adjacency matrix.
function edgeList = adj_to_edge(A)
N = length(A); %Number of nodes
edgeLinIndList = find(triu(A > 0)); %Indices of all edges
numEdges = length(edgeLinIndList);
edgeList = NaN(numEdges, 3);
for iiEdge = 1:numEdges
    [ii, jj] = ind2sub([N, N], edgeLinIndList(iiEdge)); %Node indices of edge e
    edgeList(iiEdge, :) = [ii, jj, A(ii,jj)];
end
edgeList = edgeList(randperm(numEdges), :);%Randomly order
end