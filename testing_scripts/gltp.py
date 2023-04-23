import pickle as pkl
from vedo import *
import numpy as np
import time
from os.path import dirname, abspath, join

# ----- Optional Libaraies -----
# from sklearn.neighbors import NearestNeighbors
# from pycpd import RigidRegistration
# from pycpd import AffineRegistration
# import scipy.special
# from sklearn.impute import KNNImputer

# ----- Variables -----
# Y      : a M*D matrix, centroids of the GMM. Y_0 indicates the initial centroid posiitons
# X      : a N*D matrix, input points
# beta   : a constant representing the strength of interaction between points
# G      : a M*M matrix, g_ij = exp(- abs( (y_0_i - y_0_j) / (2*beta) )**2)
#          Note: G(m, .) denotes the mth row of G
# sigma2 : variance of the GMM
# W      : a M*D matrix, matrix of Gaussian kernel weights for each point in Y
# P      : a M*N matrix, consists of posterior probabilities. 
# mu     : a constant representing how noisy the input is
# alpha  : a constant regulating the strength of smoothing
# gamma  : a constant regulating the strength of LLE
# k      : the number of nearest neighbors used in initial LLE weights calculations
# L      : a M*M matrix, consists of LLE weights
# Gi     : a k*k matrix, used for wi calculations

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

# assuming Y is sorted
# k -- going left for k indices, going right for k indices. a total of 2k neighbors.
def get_nearest_indices (k, Y, idx):
    if idx - k < 0:
        # use more neighbors from the other side?
        indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1+np.abs(idx-k)))
        # indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1))
        return indices_arr
    elif idx + k >= len(Y):
        last_index = len(Y) - 1
        # use more neighbots from the other side?
        indices_arr = np.append(np.arange(idx-k-(idx+k-last_index), idx, 1), np.arange(idx+1, last_index+1, 1))
        # indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, last_index+1, 1))
        return indices_arr
    else:
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, idx+k+1, 1))
        return indices_arr

def calc_LLE_weights (k, X):
    W = np.zeros((len(X), len(X)))
    for i in range (0, len(X)):
        indices = get_nearest_indices(int(k/2), X, i)
        xi, Xi = X[i], X[indices, :]
        component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        Gi = np.matmul(component.T, component)
        # Gi might be singular when k is large
        try:
            Gi_inv = np.linalg.inv(Gi)
        except:
            epsilon = 0.00001
            Gi_inv = np.linalg.inv(Gi + epsilon*np.identity(len(Gi)))
        wi = np.matmul(Gi_inv, np.ones((len(Xi), 1))) / np.matmul(np.matmul(np.ones(len(Xi),), Gi_inv), np.ones((len(Xi), 1)))
        W[i, indices] = np.squeeze(wi.T)

    return W

def indices_array(n):
    r = np.arange(n)
    out = np.empty((n,n,2),dtype=int)
    out[:,:,0] = r[:,None]
    out[:,:,1] = r
    return out

def cpd_lle (X, Y_0, beta, alpha, gamma, mu, max_iter=50, tol=0.00001, include_lle=True, use_geodesic=False, use_prev_sigma2=False, sigma2_0=None):

    # define params
    M = len(Y_0)
    N = len(X)
    D = len(X[0])

    # initialization
    # faster G calculation
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    converted_node_dis = []
    if not use_geodesic:
        # Gaussian Kernel
        G = np.exp(-diff / (2 * beta**2))
    else:
        # compute the geodesic distances between nodes
        seg_dis = np.sqrt(np.sum(np.square(np.diff(Y_0, axis=0)), axis=1))
        converted_node_coord = []
        last_pt = 0
        converted_node_coord.append(last_pt)
        for i in range (1, M):
            last_pt += seg_dis[i-1]
            converted_node_coord.append(last_pt)
        converted_node_coord = np.array(converted_node_coord)
        converted_node_dis = np.abs(converted_node_coord[None, :] - converted_node_coord[:, None])
        converted_node_dis_sq = np.square(converted_node_dis)

        # Gaussian Kernel
        G = np.exp(-converted_node_dis_sq / (2 * beta**2))

        # temp
        # G[converted_node_dis > 0.07] = 0
    
    Y = Y_0.copy()

    # initialize sigma2
    if not use_prev_sigma2:
        (N, D) = X.shape
        (M, _) = Y.shape
        diff = X[None, :, :] - Y[:, None, :]
        err = diff ** 2
        sigma2 = np.sum(err) / (D * M * N)
    else:
        sigma2 = sigma2_0

    # get the LLE matrix
    L = calc_LLE_weights(6, Y_0)
    H = np.matmul((np.identity(M) - L).T, np.identity(M) - L)
    
    # loop until convergence or max_iter reached
    for it in range (0, max_iter):

        # ----- E step: compute posteriori probability matrix P -----
        # faster P computation
        pts_dis_sq = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)
        c = (2 * np.pi * sigma2) ** (D / 2)
        c = c * mu / (1 - mu)
        c = c * M / N
        P = np.exp(-pts_dis_sq / (2 * sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        P = np.divide(P, den)

        max_p_nodes = np.argmax(P, axis=0)

        # if use geodesic, overwrite P
        # this section looks long, but it is simply replacing the Euclidean distances in P with geodesic distances
        if use_geodesic:
            potential_2nd_max_p_nodes_1 = max_p_nodes - 1
            potential_2nd_max_p_nodes_2 = max_p_nodes + 1
            potential_2nd_max_p_nodes_1 = np.where(potential_2nd_max_p_nodes_1 < 0, 1, potential_2nd_max_p_nodes_1)
            potential_2nd_max_p_nodes_2 = np.where(potential_2nd_max_p_nodes_2 > M-1, M-2, potential_2nd_max_p_nodes_2)
            potential_2nd_max_p_nodes_1_select = np.vstack((np.arange(0, N), potential_2nd_max_p_nodes_1)).T
            potential_2nd_max_p_nodes_2_select = np.vstack((np.arange(0, N), potential_2nd_max_p_nodes_2)).T
            potential_2nd_max_p_1 = P.T[tuple(map(tuple, potential_2nd_max_p_nodes_1_select.T))]
            potential_2nd_max_p_2 = P.T[tuple(map(tuple, potential_2nd_max_p_nodes_2_select.T))]
            next_max_p_nodes = np.where(potential_2nd_max_p_1 > potential_2nd_max_p_2, potential_2nd_max_p_nodes_1, potential_2nd_max_p_nodes_2)
            node_indices_diff = max_p_nodes - next_max_p_nodes
            max_node_smaller_index = np.arange(0, N)[node_indices_diff < 0]
            max_node_larger_index = np.arange(0, N)[node_indices_diff > 0]
            dis_to_max_p_nodes = np.sqrt(np.sum(np.square(Y[max_p_nodes]-X), axis=1))
            dis_to_2nd_largest_p_nodes = np.sqrt(np.sum(np.square(Y[next_max_p_nodes]-X), axis=1))
            converted_P = np.zeros((M, N)).T

            for idx in max_node_smaller_index:
                converted_P[idx, 0:max_p_nodes[idx]+1] = converted_node_dis[max_p_nodes[idx], 0:max_p_nodes[idx]+1] + dis_to_max_p_nodes[idx]
                converted_P[idx, next_max_p_nodes[idx]:M] = converted_node_dis[next_max_p_nodes[idx], next_max_p_nodes[idx]:M] + dis_to_2nd_largest_p_nodes[idx]

            for idx in max_node_larger_index:
                converted_P[idx, 0:next_max_p_nodes[idx]+1] = converted_node_dis[next_max_p_nodes[idx], 0:next_max_p_nodes[idx]+1] + dis_to_2nd_largest_p_nodes[idx]
                converted_P[idx, max_p_nodes[idx]:M] = converted_node_dis[max_p_nodes[idx], max_p_nodes[idx]:M] + dis_to_max_p_nodes[idx]

            converted_P = converted_P.T

            P = np.exp(-np.square(converted_P) / (2 * sigma2))
            den = np.sum(P, axis=0)
            den = np.tile(den, (M, 1))
            den[den == 0] = np.finfo(float).eps
            c = (2 * np.pi * sigma2) ** (D / 2)
            c = c * mu / (1 - mu)
            c = c * M / N
            den += c

            P = np.divide(P, den)

        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        PX = np.matmul(P, X)

        print(Pt1)
    
        # ----- M step: solve for new weights and variance -----
        if include_lle:
            A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 * gamma * np.matmul(H, G)
            B_matrix = PX - np.matmul(np.diag(P1) + sigma2*gamma*H, Y_0)
        else:
            A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M)
            B_matrix = PX - np.matmul(np.diag(P1), Y_0)

        # solve for W
        W = np.linalg.solve(A_matrix, B_matrix)

        T = Y_0 + np.matmul(G, W)
        trXtdPt1X = np.trace(np.matmul(np.matmul(X.T, np.diag(Pt1)), X))
        trPXtT = np.trace(np.matmul(PX.T, T))
        trTtdP1T = np.trace(np.matmul(np.matmul(T.T, np.diag(P1)), T))

        # solve for sigma^2
        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D)

        # update Y
        if pt2pt_dis_sq(Y, Y_0 + np.matmul(G, W)) < tol:
            # if converged, break loop
            Y = Y_0 + np.matmul(G, W)
            print("iteration until convergence:", it)
            break
        else:
            # keep going until max iteration is reached
            Y = Y_0 + np.matmul(G, W)

            if it == max_iter - 1:
                print("did not converge!")
    
    return Y, sigma2

if __name__ == "__main__":

    # load recorded data
    # load X_0 (wire point cloud - initial frame)
    data_dir = dir = join(dirname(abspath(__file__)), "data/frames/")
    f = open(data_dir + '000_pcl.json', 'rb')
    X_0 = pkl.load(f, encoding="bytes")
    f.close()

    # downsample the dense wire point cloud
    X = X_0.copy()
    X = np.array(X)
    X = X[::int(1/0.05)]

    # create occlusion
    # X = X[X[:, 0] < 0.12]
    # X = X[X[:, 0] > -0.15]

    # load Y_0 (sorted) and sigma2_0 (GMM centroids and variance - initial frame)
    f = open(data_dir + 'nodes/000_nodes.json', 'rb')
    data = pkl.load(f, encoding="bytes")
    f.close()
    Y_0, sigma2_0 = np.array(data[0]), data[1]

    # tracking params
    beta = 0.5                 # beta   : a constant representing the strength of interaction between points
    alpha = 5                  # alpha  : a constant regulating the strength of MCT smoothing
    gamma = 1                  # gamma  : a constant regulating the strength of LLE
    mu = 0.9                   # mu     : a constant representing how noisy the input point cloud is (0 - 1)
    max_iter = 30
    tol = 0.00001
    include_lle = True
    use_geodesic = True
    use_prev_sigma2 = True

    cur_time = time.time()
    Y, sigma2 = cpd_lle(X = X, 
                        Y_0 = Y_0, 
                        beta = beta,       # beta   : a constant representing the strength of interaction between points
                        alpha = alpha,     # alpha  : a constant regulating the strength of smoothing
                        gamma = gamma,     # gamma  : a constant regulating the strength of LLE
                        mu = mu,           # mu     : a constant representing how noisy the input is (0 - 1)
                        max_iter = max_iter, 
                        tol = tol, 
                        include_lle = include_lle, 
                        use_geodesic = use_geodesic, 
                        use_prev_sigma2 = use_prev_sigma2, 
                        sigma2_0 = sigma2_0)
    
    print("time taken = ", time.time() - cur_time)

    plt = Plotter(N=1, axes=0)
    pc = Points(X, c=(0, 0, 0), r=4)
    pc_no_occlusion = Points(X_0[::int(1/0.05)], c=(0, 255, 0), r=4)

    nodes_orig = Points(Y_0, c=(0, 0, 255), r=15)
    nodes_orig_lines = DashedLine(Y_0, c=(0, 0, 255), lw=3)
    nodes = Points(Y, c=(255, 0, 0), r=15)
    nodes_lines = DashedLine(Y, c=(255, 0, 0), lw=3)

    plt.show(pc_no_occlusion, pc, nodes, nodes_orig, nodes_lines, nodes_orig_lines, at=0)
    plt.interactive().close()

    # more frames?
    num_of_frames = 10
    Y_last = Y[0:25].copy()
    for i in range (2, num_of_frames):  # the next frame is frame 2
        sample_prefix = ''
        if len(str(i)) == 1:
            sample_prefix = '00'
        elif len(str(i)) == 2:
            sample_prefix ='0'
        else:
            sample_prefix = ''
        sample_id = sample_prefix + str(i)

        f = open(data_dir + '001' + '_pcl.json', 'rb')
        X_0 = pkl.load(f, encoding="bytes")
        f.close()

        X = X_0.copy()
        X = np.array(X)
        X = X[::int(1/0.05)]

        # create occlusion
        # X = X[X[:, 0] < 0.12]
        # X = X[X[:, 0] > -0.15]
        X = X[~((0 < X[:, 0]) & (X[:, 0] < 0.1) & (X[:, 1] > 0))]

        cur_time = time.time()
        Y, sigma2 = cpd_lle(X = X, 
                            Y_0 = Y_last, 
                            beta = beta,       # beta   : a constant representing the strength of interaction between points
                            alpha = alpha,     # alpha  : a constant regulating the strength of smoothing
                            gamma = gamma,     # gamma  : a constant regulating the strength of LLE
                            mu = mu,           # mu     : a constant representing how noisy the input is (0 - 1)
                            max_iter = max_iter, 
                            tol = tol, 
                            include_lle = include_lle, 
                            use_geodesic = use_geodesic, 
                            use_prev_sigma2 = use_prev_sigma2, 
                            sigma2_0 = sigma2_0)
        
        print("time taken = ", time.time() - cur_time)

        plt = Plotter(N=1, axes=0)
        pc = Points(X, c=(0, 0, 0), r=4)
        pc_no_occlusion = Points(X_0[::int(1/0.05)], c=(0, 255, 0), r=4)

        nodes_orig = Points(Y_last, c=(0, 0, 255), r=15)
        nodes_orig_lines = DashedLine(Y_last, c=(0, 0, 255), lw=4)
        nodes = Points(Y, c=(255, 0, 0), r=15)
        nodes_lines = DashedLine(Y, c=(255, 0, 0), lw=4)

        Y_last = Y.copy()

        plt.show(pc_no_occlusion, pc, nodes, nodes_orig, nodes_lines, nodes_orig_lines, at=0)
        plt.interactive().close()