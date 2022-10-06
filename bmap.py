from layouts import optimize_layout_euclidean
# from umap.layouts import (
#     optimize_layout_euclidean,
#     optimize_layout_generic,
#     optimize_layout_inverse,
# )
from scipy.spatial.distance import cdist
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans, KMeans
import numba
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle

import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
                          csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore', SparseEfficiencyWarning)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

def cdist_leveshtein_tqmd(sequences, index_x, index_y):

    from Levenshtein import distance
    import numpy as np

    # estimate time cost 1000x1000 = 8.7s
    n = len(index_x)
    m = len(index_y)
    distance_matrix = np.zeros((n, m), dtype=np.int32)
    # n_entry = round((n*m - max(n, m))/2)
    # with tqdm(total=n_entry) as progress_bar:
    indicator = {}
    for i, idx_i in enumerate(tqdm(index_x)):
        for j, idx_j in enumerate(index_y):
            if idx_i == idx_j:
                continue
            mark = (min(idx_j, idx_i), max(idx_j, idx_i))
            if mark in indicator:
                distance_matrix[i, j] = indicator[mark]
                continue
            distance_matrix[i, j] = distance(
                sequences[idx_i], sequences[idx_j])
            indicator[mark] = distance_matrix[i, j]

    return distance_matrix


def get_distance_matrix(sequences_A, sequences_B):
    # sequences is a numpy array that can be obtained from df.seq.values

    from scipy.spatial.distance import cdist
    from Levenshtein import distance

    transformed_strings_A = sequences_A.reshape(-1, 1)
    transformed_strings_B = sequences_B.reshape(-1, 1)

    distance_matrix = cdist(
        transformed_strings_A, transformed_strings_B,
        lambda x, y: distance(x[0], y[0]))
    return distance_matrix


def get_landmarks(X, n_landmarks, mode="random", random_state=42, unit=10,
                  datatype="numeric"):

    if mode == "random":
        # print("random based landmark selection")
        from sklearn.utils import shuffle
        n = len(X)
        index = range(n)
        index = shuffle(index, random_state=random_state)
        landmarks_index = index[:n_landmarks]
        return X[landmarks_index], landmarks_index

    elif mode == "kmeans":
        if datatype == "numeric":
            print("MiniBatchKMeans based landmark selection")
            kmeans = MiniBatchKMeans(n_clusters=n_landmarks, n_init=1,
                                     max_no_improvement=1, init="random",
                                     random_state=random_state,
                                     ).fit(X)
            return kmeans.cluster_centers_, kmeans.labels_
        else:
            print("large-scale spectral clustering")
            n_landmarks_tmp = 100
            n_seq = len(X)
            index_x, index_y = list(range(n_seq)), list(range(n_landmarks_tmp))
            distance_matrix_tmp = cdist_leveshtein_tqmd(X, index_x, index_y)
            emb = bipartiteSpectralEmbedding(
                distance_matrix_tmp, n_landmarks_tmp//10)
            kmeans = MiniBatchKMeans(n_clusters=n_landmarks, n_init=1,
                                     max_no_improvement=1, init="random",
                                     random_state=random_state).fit(emb)
            index_landmark = []
            Nx = len(X)
            for i in np.unique(kmeans.labels_):
                idx = kmeans.labels_ == i
                dist_tmp = distance_matrix_tmp[idx]
                idx_min = dist_tmp.sum(1).argmin()
                index_min = np.arange(Nx)[idx][idx_min]
                index_landmark.append(index_min)

            index_landmark = np.array(index_landmark)
            LdFea = X[index_landmark]
            return LdFea, kmeans.labels_

    elif mode == 'DnC':
        print("divide-and-conquer based landmark selection")
        selectN = 10000
        unit = 10
        [label, centers] = DnC_landmark(X, n_landmarks, selectN, unit)
        return centers, label


def set_sub_k(p):
    if len(p) == 1:
        return p.astype(int)

    intp = np.ceil(p)
    sumright = round(sum(p))
    intp[intp < 1] = 1
    diff = sum(intp) - sumright

    while diff > 0:
        idx = intp > 1
        u = min(intp[idx])
        idx_u = intp == u
        n = int(sum(idx_u))

        if diff < (u - 1) * n:
            # select ni in samples
            ni = np.floor(diff / (u-1)).astype(int)
            randSample = np.zeros((n,), dtype=bool)
            randSample[np.random.choice(n, ni, replace=False)] = 1
            selectedIdx, restIdx = idx_u.copy(), idx_u.copy()
            selectedIdx[idx_u], restIdx[idx_u] = randSample, ~randSample
            # select one in rest idx
            randSample = np.zeros((n-ni,), dtype=bool)
            randSample[np.random.choice(n-ni)] = 1
            restIdx[restIdx] = 1
            intp[restIdx] = intp[restIdx] - (diff - ni * (u-1))

            diff = 0
        else:
            intp[idx_u] = 1
            diff = diff - (u-1) * n

    if diff < 0:
        print("wrong set_sub_k!")

    return intp.astype(int)


def DnC_landmark(fea, p, selectN, unit):
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    [n, d] = fea.shape
    maxIter = 10
    centers = np.zeros((p, d))
    sumD = np.ones((p,))
    obtainedSubsets = 1
    label = np.zeros((n,), dtype=int)
    while obtainedSubsets < p:
        labelNew = label
        ks = sumD[:obtainedSubsets]
        ks = set_sub_k(ks/sum(ks) * p)
        ks[ks > unit] = unit
        for i in range(obtainedSubsets):
            k = ks[i]
            if k == 1:
                continue

            indi = label == i
            curN = sum(indi)

            if curN > selectN:
                # light-k-means
                # random sampling
                randSample = np.zeros((curN,), dtype=bool)
                randSample[np.random.choice(curN, selectN, replace=False)] = 1
                selectedIdx, restIdx = indi.copy(), indi.copy()
                selectedIdx[indi], restIdx[indi] = randSample, ~randSample

                # obtain centers
                kmeans = MiniBatchKMeans(n_clusters=k, max_iter=maxIter,
                                         random_state=5).fit(fea[selectedIdx])
                curLabel = kmeans.labels_
                curSumD = kmeans.counts_
                curCenter = kmeans.cluster_centers_
                # insert the rest samples
                D = cdist(fea[restIdx], curCenter)
                RestD, restLabel = np.min(D, axis=1), np.argmin(D, axis=1)
                curLabel = np.zeros((curN,))
                curLabel[randSample] = kmeans.labels_
                curLabel[~randSample] = restLabel
                # add the rest SumD
                if sum(ks) < p:
                    restN = sum(restIdx)
                    curSumD = curSumD + RestD.T * \
                        csr_matrix((np.ones(restN,), restLabel,
                                   np.arange(restN+1)), shape=(restN, k))
            elif sum(indi) <= k:
                k = min(sum(indi), 2)
                kmeans = MiniBatchKMeans(n_clusters=k, max_iter=maxIter,
                                         random_state=5).fit(fea[indi])
                curLabel = kmeans.labels_
                curSumD = kmeans.counts_
                curCenter = kmeans.cluster_centers_
            else:
                kmeans = MiniBatchKMeans(n_clusters=k, max_iter=maxIter,
                                         random_state=5).fit(fea[indi])
                curLabel = kmeans.labels_
                curSumD = kmeans.counts_
                curCenter = kmeans.cluster_centers_
            # modify the local label and insert it into global one
            remainLabelIdx = curLabel == 0
            curLabel[remainLabelIdx] = i
            curLabel[~remainLabelIdx] = curLabel[~remainLabelIdx] + \
                obtainedSubsets - 1
            labelNew[indi] = curLabel
            # insert centers and sumD into global one
            centers[i] = curCenter[0]
            idx_rest = np.arange(obtainedSubsets, obtainedSubsets + k - 1)
            centers[idx_rest] = curCenter[1:]
            sumD[i] = curSumD[0]
            sumD[idx_rest] = curSumD[1:]
            # count obtained clusters
            obtainedSubsets = obtainedSubsets + k - 1

        label = labelNew
    return label, centers


def fast_knn(X, n_neighbors=10):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    knn_distances = np.empty((X.shape[0], n_neighbors))
    for row in range(X.shape[0]):
        knn_indices[row] = np.argpartition(X[row], n_neighbors)[:n_neighbors]
        knn_distances[row] = X[row, knn_indices[row]]
    return knn_distances, knn_indices


def myKNN_dist(fea, RpFea, Knn, fea_idx, RpFeaKnnIdx, datatype="numeric"):
    p = max(fea_idx)+1
    n = fea.shape[0]
    RpFeaKnnDist = np.empty((n, Knn))
    RpFeaKnnIdxFull = RpFeaKnnIdx[fea_idx]
    for i in numba.prange(p):
        tmp = fea_idx == i
        if datatype == "numeric":
            RpFeaKnnDist[tmp] = cdist(fea[tmp], RpFea[RpFeaKnnIdx[i], :])
        else:
            RpFeaKnnDist[tmp] = get_distance_matrix(
                fea[tmp], RpFea[RpFeaKnnIdx[i]])
    return RpFeaKnnDist, RpFeaKnnIdxFull


def get_knn_distance(fea, LdFea, fea_idx, knn=5, min_dist=0.2, datatype="numeric",
                     metric=None):

    Nx = fea.shape[0]
    Ny = LdFea.shape[0]

    Knn = 10 * knn
    # # distance matrix of landmarks
    if datatype == "numeric":
        LdFeaW = cdist(LdFea, LdFea)
    else:
        LdFeaW = get_distance_matrix(LdFea, LdFea)
    # modified to avoid self distance 0
    LdFeaW = LdFeaW + 1e100 * np.eye(LdFeaW.shape[0])

    # find Knn of landmark
    [LdKnnDist, LdKnnIdx] = fast_knn(LdFeaW, Knn)
    # find -Knn of landmarks
    # [_, LdNegKnnIdx] = fast_knn(LdFeaW, int(Ny*0.3))

    # find distance matrix between samples and their possible nearest neighbors
    [LdFeaKnnDist, LdFeaKnnIdxFull] = myKNN_dist(fea, LdFea,
                                                    Knn, fea_idx, LdKnnIdx, datatype)
    # find knn of neighbors according to possible landmark distances
    [knnDist, LdKnnIdx_tmp] = fast_knn(LdFeaKnnDist, knn)
    knnIdx = np.take_along_axis(LdFeaKnnIdxFull, LdKnnIdx_tmp, axis=1)

    # build sparse matrix
    indices = knnIdx.flatten()
    indptr = np.arange(0, (Nx+1) * knn, knn)
    distances = knnDist.flatten()
    # return indices, indptr, distances
    mean_dist = np.mean(distances)
    distances = np.exp(-distances**2/(mean_dist**2))
    distances[distances > (1-min_dist)] = 1
    B = csr_matrix((distances, indices, indptr), shape=(Nx, Ny))

    return B, mean_dist


def bipartiteSpectralEmbedding(B, dim=10, add_landmarks=False, random_state=42):
    # return embedding
    [Nx, Ny] = B.shape

    # 2. transport cuts

    # normalized Z
    # [Nx, Ny] = B.shape
    dx = B.sum(1)
    dx[dx == 0] = 1e-10  # Just to make 1./dx feasible.
    Dx = csr_matrix((1/np.array(dx).flatten(),
                     (np.arange(Nx), np.arange(Nx))), shape=(Nx, Nx))
    try:
        Wy = B.T * Dx * B
    except:
        Wy = np.matmul(B.T, (Dx * B))

    # % compute Ncut eigenvectors
    # % normalized affinity matrix
    d = Wy.sum(1)
    d[d == 0] = 1e-10  # Just to make 1./dx feasible.
    D = csr_matrix(
        (1/np.sqrt(np.array(d).flatten()), (np.arange(Ny), np.arange(Ny))), shape=(Ny, Ny))

    nWy = D * Wy * D
    nWy = (nWy + nWy.T) / 2
    # % computer eigenvectors
    k = dim + 1

    eval, evec = eigsh(nWy, k)
    eval, evec = eval[::-1], evec[:, ::-1]
    eval, evec = eval[1:], evec[:, 1:]
    try:
        embedding = Dx * B * evec
    except:
        embedding = np.matmul(Dx * B, evec)

    if add_landmarks:
        embedding = np.concatenate([embedding, evec], axis=0)

    #  normalize each row to unit norm

    embedding = embedding / np.sqrt((embedding ** 2).sum(0) + 1e-10)

    return embedding



@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho

@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    return_dists=False,
    bipartite=False,
):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    return_dists: bool (optional, default False)
        Whether to return the pairwise distance associated with each edge

    bipartite: bool (optional, default False)
        Does the nearest neighbour set represent a bipartite graph?  That is are the
        nearest neighbour indices from the same point set as the row indices?

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)

    dists: array of shape (n_samples * n_neighbors)
        Distance associated with each entry in the resulting sparse matrix
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def large_scale_ensemble_spectral_clustering(X, n=10, n_landmarks=100,
                                             cluster_size=[10, 100], random_state=42):
    get_landmarks(X, n_landmarks, "kmeans", random_state)
    pass


def find_ab_params(spread, min_dist):
    from scipy.optimize import curve_fit
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    # print(params[0], params[1])
    return params[0], params[1]


class BMAP():
    def __init__(self, n_components=2, random_state=42,
                 n_neighbors_a=10, n_neighbors_b=10,
                 n_landmarks=10000, n_epochs=300, datatype="numeric",
                 min_dist=0.1, spread=1,
                 ax=3, bx=1,
                 ay=3, by=1,
                 init = "random",
                 verbose= True,
                 parallel = True,
                 stored_names="bmap_variables"):
        self.n_components = n_components
        self.n_neighbors_a = n_neighbors_a
        self.n_neighbors_b = n_neighbors_b
        self.random_state = random_state
        self.n_landmarks = n_landmarks
        self.n_epochs = n_epochs
        self.datatype = datatype
        self.state = 0
        self.stored_names = stored_names
        self.init = init
        self.min_dist = min_dist
        self.spread = spread
        self.parallel = parallel
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        # self.a, self.b = find_ab_params(spread, min_dist)
        # self.ax, self.bx = self.a, self.b
        # self.ay, self.by = self.a, self.b
        self.verbose = verbose

    def fit_landmarks(self, fea):
        # compute landmark
        if self.verbose:
            print("compute landmark")
        if self.init == "kmeans":
            self.LdFea, self.fea_idx = get_landmarks(
                fea, self.n_landmarks, mode="kmeans",
                random_state=self.random_state,
                datatype=self.datatype)
        elif self.init == "random":
            if (self.n_landmarks == fea.shape[0]) | (self.n_landmarks <= 0):
                self.LdFea, self.landmarks_index = fea, np.arange(fea.shape[0])
                self.n_landmarks = fea.shape[0]
            elif (self.n_landmarks > 0) & (self.n_landmarks < fea.shape[0]):
                self.LdFea, self.landmarks_index = get_landmarks(
                    fea, self.n_landmarks, mode="random",
                    random_state=self.random_state,
                    datatype=self.datatype)
            else:
                raise("n_landmark is out of range!")
            

    def fit_graph(self, fea):
        if self.verbose:
            print("compute bipartite knn graph")

        if self.init == "kmeans":
            # compute bipartite knn graph
            self.B, self.mean_dist = get_knn_distance(
                fea, self.LdFea, self.fea_idx, self.n_neighbors_b, datatype=self.datatype)
            self.B.eliminate_zeros()
            self.graph_B = self.B.tocoo()
            self.graph_B.sum_duplicates()

            # graph A is Knn for landmark side
            A = np.ones((self.n_landmarks, fea.shape[0])) * 1e100
            for cluster in numba.prange(self.n_landmarks):
                cluster_idx = self.fea_idx == cluster
                if self.datatype == "numeric":
                    A[cluster, cluster_idx] = cdist(
                        fea[cluster_idx], self.LdFea[cluster].reshape((1, -1))).flatten()
                else:
                    A[cluster, cluster_idx] = get_distance_matrix(
                        fea[cluster_idx], np.array(self.LdFea[cluster])).flatten()

            [knnDist, KnnIdx] = fast_knn(A, self.n_neighbors_a)
            indices = KnnIdx.flatten()
            indptr = np.arange(0, (self.n_landmarks+1) *
                            self.n_neighbors_a, self.n_neighbors_a)
            distances = knnDist.flatten() + 1e-10
            # distances = np.exp(-distances**2/(np.mean(distances)**2))
            distances = np.exp(-distances**2/(self.mean_dist**2))
            self.A = csr_matrix((distances, indices, indptr),
                                shape=(self.n_landmarks, fea.shape[0]))
            self.A.eliminate_zeros()
            self.graph_A = self.A.tocoo()
            self.graph_A.sum_duplicates()
        
        elif self.init == "random":
            # knn = self.n_neighbors_b

            # import hnswlib
            # dim = fea.shape[1]
            # num_elements = self.n_landmarks
            # p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
            # p.init_index(max_elements=num_elements, ef_construction=100, M=16)
            # p.add_items(self.LdFea)
            # knnIdx, knnDist = p.knn_query(fea, k=self.n_neighbors_b+1)

            import pynndescent
            index = pynndescent.NNDescent(self.LdFea, n_neighbors=self.n_neighbors_a+1,
                 random_state = self.random_state)
            if self.n_landmarks == fea.shape[0]:
                knnIdx, knnDist = index.neighbor_graph
            else:
                neighbors = index.query(fea, k = self.n_neighbors_b+1)
                knnIdx = neighbors[0]
                knnDist = neighbors[1]
            # avoid self neighbors
            idx_0_neighbors = knnDist[:,0] == 0
            knnIdx[idx_0_neighbors,:self.n_neighbors_b] = knnIdx[idx_0_neighbors,1:self.n_neighbors_b+1]
            knnDist[idx_0_neighbors,:self.n_neighbors_b] = knnDist[idx_0_neighbors,1:self.n_neighbors_b+1]
            knnIdx = knnIdx[:,:self.n_neighbors_b]
            knnDist = knnDist[:,:self.n_neighbors_b]
            # knn = knnIdx.shape[1]
            # print(knn)
            
            # indices = knnIdx.flatten()
            knnDist = knnDist.astype(np.float32)
            self.knnIdx = knnIdx
            
            local_connectivity = 1.0
            sigmas, rhos = smooth_knn_dist(
                knnDist,
                float(self.n_neighbors_b),
                local_connectivity=float(local_connectivity),
            )
            rows, cols, vals, dists = compute_membership_strengths(
                knnIdx, knnDist, sigmas, rhos
            )

            self.B = scipy.sparse.coo_matrix(
                (vals, (rows, cols)), shape=(fea.shape[0], self.n_landmarks)
            )

            # Nx = fea.shape[0]
            # Ny = self.n_landmarks
            # indptr = np.arange(0, (Nx+1) * knn, knn)
            # distances = knnDist.flatten()
            # # return indices, indptr, distances
            # mean_dist = np.mean(distances)
            # distances = np.exp(-distances**2/(mean_dist**2))
            # distances[distances > (1-self.min_dist)] = 1
            # self.B = csr_matrix((distances, indices, indptr), shape=(Nx, Ny))
            self.B.eliminate_zeros()
            self.graph_B = self.B.tocoo()
            self.graph_B.sum_duplicates()

            # # graph A is landmarks knn 
            # knn_indices, knn_dists = index.neighbor_graph
            # indices = knn_indices.flatten()
            # knn = self.n_neighbors_a
            # indptr = np.arange(0, (Ny+1) * knn, knn)
            # distances = knn_dists.flatten()
            # # return indices, indptr, distances
            # mean_dist = np.mean(distances)
            # distances = np.exp(-distances**2/(mean_dist**2))
            # distances[distances > (1-self.min_dist)] = 1
            # self.A = csr_matrix((distances, indices, indptr), shape=(Ny, Ny))
            # self.A.eliminate_zeros()
            # self.graph_A = self.A.tocoo()
            # self.graph_A.sum_duplicates()

    def fit_init_embedding(self):
        # compute spectral embedding
        if self.verbose:
            print("compute spectral embedding")
        if self.init == "kmeans":
            add_landmarks = True
        elif self.init == "random":
            add_landmarks = False
        self.init_emb = bipartiteSpectralEmbedding(
            self.B, self.n_components, add_landmarks=add_landmarks)
        

    def fit_optimize(self):
        # add noise to embedding to avoid local minimal
        expansion = 10.0 / np.abs(self.init_emb).max()
        self.embedding = (self.init_emb * expansion).astype(
            np.float32
        ) + self.random_state_checked.normal(
            scale=0.0001, size=[self.init_emb.shape[0], self.n_components]
        ).astype(
            np.float32
        )
        # 首先对于全局相似度的计算存在问题，因为A，B矩阵的距离系数使用不一，所以有问题

        self.graph_B.data[self.graph_B.data < (
            self.graph_B.data.max() / self.n_epochs)] = 0.0

        self.graph_B.eliminate_zeros()
        self.graph_B = self.B.tocoo()
        self.graph_B.sum_duplicates()

        n_vertices = self.graph_B.shape[0]
        n_landmarks = self.graph_B.shape[1]
        if self.init =="random":
            head = self.graph_B.row
            tail = np.array(self.landmarks_index)[self.graph_B.col]
            # head2 = np.array(self.landmarks_index)[self.graph_A.row]
            # tail2 = np.array(self.landmarks_index)[self.graph_A.col]

            # head_all = np.concatenate([tail2, tail, head, head2])
            # tail_all = np.concatenate([head2, head, tail, tail2])
            # if n_landmarks == n_vertices:
            #     head_all = head
            #     tail_all = tail
            # else:    
            head_all = np.concatenate([tail, head])
            tail_all = np.concatenate([head, tail])

            head_all = shuffle(head_all, random_state = 42)
            tail_all = shuffle(tail_all, random_state = 42)
        else:
            head = self.graph_B.row
            tail = self.graph_B.col + n_vertices
            head2 = self.graph_A.row + n_vertices
            tail2 = self.graph_A.col
            head_all = np.concatenate([head2, head])
            tail_all = np.concatenate([tail2, tail])

        # b = np.ones((n_vertices,))* self.b

        # a = np.concatenate([np.ones((n_vertices,))* self.a, np.ones((n_landmarks,))* self.b])
        # b = np.concatenate([np.ones((n_vertices,))* self.a, np.ones((n_landmarks,))* self.b])

        # head_all = np.concatenate([head2, head, tail2, tail])
        # tail_all = np.concatenate([tail2, tail, head2, head])


        rng_state = self.random_state_checked.randint(
            INT32_MIN, INT32_MAX, 3).astype(np.int64)
        epochs_per_sample = make_epochs_per_sample(
            self.graph_B.data, self.n_epochs)

        # optimize layout
        if self.verbose:
            print("optimize layout")
        self.embedding = optimize_layout_euclidean(
            self.embedding,
            self.embedding,
            head_all,
            tail_all,
            n_epochs=self.n_epochs,
            n_vertices=n_vertices,
            epochs_per_sample=epochs_per_sample,
            ax=self.ax,
            bx=self.bx,
            ay=self.ay,
            by=self.by,
            rng_state=rng_state,
            gamma=1,
            initial_alpha=1,
            negative_sample_rate=5,
            # negative_landmarks=self.neg_nearest_landmark,
            parallel=self.parallel,
        )

    def load_stored_variables(self):
        import os
        if os.path.exists(self.stored_names):
            print("load_stored_variables")
            file_to_read = open(self.stored_names, "rb")
            stored_BMAP = pickle.load(file_to_read)
            file_to_read.close()
            for attr in dir(stored_BMAP):
                tmp = getattr(stored_BMAP, attr)
                if type(tmp).__module__ == np.__name__ or type(getattr(stored_BMAP, attr)) == int:
                    setattr(self, attr, tmp)
            print("state %d" % self.state)

    def save_stored_variables(self):
        file_to_store = open(self.stored_names, "wb")
        pickle.dump(self, file_to_store)
        file_to_store.close()

    def adjust_embedding(self):
        if self.verbose:
            print("adjusting embedding")
        # adjust obvious outliers
        maxp = 99.9
        embedding = self.embedding
        for i in range(self.n_components):
            maxx = np.percentile(embedding[:,i], maxp)
            embedding[embedding[:,i] > maxx,i] = maxx
            minx = np.percentile(embedding[:,i], 100 - maxp)
            embedding[embedding[:,i] < minx,i] = minx

        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(n_estimators = 10, max_samples=0.99)
        clf.fit(embedding)  # fit 10 trees  
        y_pred_outliers = clf.predict(embedding)

        outlier_idx = y_pred_outliers==-1
        # normal_idx = y_pred_outliers==1
        # plt.scatter(embedding[idx,0], embedding[idx,1], alpha= 0.2)
        # embedding_outlier = embedding[np.array(bmap.landmarks_index)[bmap.knnIdx[outlier_idx, 1]]]
        # plt.scatter(embedding[idx,0], embedding[idx,1], alpha= 0.2)
        # plt.legend(["1","-1"])
        # plt.show()
        
        outlier_knn_idx1 = np.array(self.landmarks_index)[self.knnIdx[outlier_idx, 0]]
        outlier_knn_idx2 = np.array(self.landmarks_index)[self.knnIdx[outlier_idx, 1]]
        
        embedding[outlier_idx] = (embedding[outlier_knn_idx1]*.999 + embedding[outlier_knn_idx2]*0.001)
        

    def fit(self, fea, y=None):
        self.random_state_checked = check_random_state(self.random_state)
        # self.load_stored_variables()

        # if self.state == 0:
        self.fit_landmarks(fea)
        # self.state = 1
        # self.save_stored_variables()
        # if self.state == 1:
        self.fit_graph(fea)
        # self.state = 2
        # self.save_stored_variables()
        # if self.state == 2:
        self.fit_init_embedding()
        # self.state = 3
        # self.save_stored_variables()
        # if self.state == 3:
        import gc
        gc.collect()
        self.fit_optimize()
        self.adjust_embedding()
        # self.state = 4
        # self.save_stored_variables()
        return self

    def fit_transform(self, fea):
        self = self.fit(fea)
        embedding = self.embedding[:fea.shape[0]]
        return embedding
