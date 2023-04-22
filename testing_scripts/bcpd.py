import pickle as pkl
from vedo import *
import numpy as np
import time
from os.path import dirname, abspath, join
import scipy

# ===== Notations =====
# c = (c_1, ..., c_N)^T \in {0, 1}^N
#   -- the vector of indicator variables where the nth element c_n takes 1 if the nth target point is a non-outlier and 0 otherwise
# v = (v_1^T, ..., v_M^T)^T 
#   -- the displacement vectors
# e = (e_1, ..., e_N)^T \in {0, ..., M}^N
#   -- the vector of index variables where the nth element e_n indicates the index of a source point that corresponds to x_n, e_n = m means x_n corresponds to y_m
# rho = (s, R, t) 
#   -- the set of random variables that defines a similarity transformation T
# alpha = (alpha_1, ..., alpha_M)^T \in [0, 1]^M
#   -- the vector of probabilities that satisfies \sum_{m=1}^M alpha_m = 1, where alpha_m represents the probability of an event e_n = m for any n
# nu = (nu_1, ..., nu_M)^T \in R^M with nu_m = \sum_{n=1}^N p_mn
#   -- the estimated numbers of target points matched with each source point
# nu_prime = (nu_prime_1, ..., nu_prime_n)^T \in R^N with nu_prime_n = \sum_{m=1}^m p_mn
#   -- the vector of probabilities where nu_prime_n is the posterior probability that the nth target point is a non-outlier

# phi_mn          -- phi(x_n; T(y_m), sigma^2)
# psi(.)          -- the digamma function
# G_tilde         -- Kronecker(G, I_D)
# P_tilde         -- Kronecker(P, I_D)
# nu_tilde        -- Kronecker(nu, 1_D)
# nu_prime_tilde  -- Kronecker(nu_prime, 1_D)

# ===== Parameters =====
# X \in R^N  -- target point set
# Y \in R^M  -- source point set 
# omega      -- the outlier probability
# kappa      -- the parameter of the Dirichlet distribution used as a prior distribution of alpha
# gamma      -- the scale factor of sigma2_0
# beta       -- controls the influence of motion coherence

def bcpd (X, Y, beta, omega, lam, kappa, gamma, max_iter = 50, tol = 0.00001, sigma2_0 = None):
    # ===== initialization =====
    N = len(X)
    M = len(Y)

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    Y_hat = Y.copy()
    v_hat = np.zeros((M, 3))

    big_sigma = np.eye(M)
    alpha_m_bracket = np.ones((M, N)) * 1.0/M
    s = 1
    R = np.eye(3)
    t = np.zeros((3,))

    # initialize G
    diff = Y[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    G = np.exp(-diff / (2 * beta**2))

    # # geodesic distance
    # seg_dis = np.sqrt(np.sum(np.square(np.diff(Y, axis=0)), axis=1))
    # converted_node_coord = []
    # last_pt = 0
    # converted_node_coord.append(last_pt)
    # for i in range (1, M):
    #     last_pt += seg_dis[i-1]
    #     converted_node_coord.append(last_pt)
    # converted_node_coord = np.array(converted_node_coord)
    # converted_node_dis = np.abs(converted_node_coord[None, :] - converted_node_coord[:, None])
    # converted_node_dis_sq = np.square(converted_node_dis)
    # G = 0.9 * np.exp(-converted_node_dis_sq / (2 * beta**2)) + 0.1 * G

    # # G approximation
    # eigen_values, eigen_vectors = np.linalg.eig(G)
    # positive_indices = eigen_values > 0
    # G_hat = eigen_vectors[:, positive_indices] @ np.diag(eigen_values[positive_indices]) @ eigen_vectors[:, positive_indices].T
    # # print(eigen_values.astype(np.float64))
    # # print(type(G_hat[0, 0]))
    # # return
    # G = G_hat.astype(np.float64)

    # initialize sigma2
    if sigma2_0 is None:
        diff = X[None, :, :] - Y[:, None, :]
        err = diff ** 2
        sigma2 = gamma * np.sum(err) / (3 * M * N)
    else:
        sigma2 = gamma * sigma2_0

    # ===== log time and initial values =====
    start_time = time.time()
    prev_Y_hat = Y_hat.copy()
    prev_sigma2 = sigma2

    for i in range (0, max_iter):
        Y_hat_flat = Y_hat.flatten()
        v_hat_flat = v_hat.flatten()

        # ===== update P and related terms =====
        pts_dis_sq = np.sum((X[None, :, :] - Y_hat[:, None, :]) ** 2, axis=2)
        P = np.exp(-pts_dis_sq / (2 * sigma2))
        P *= alpha_m_bracket

        c = (2 * np.pi * sigma2) ** (3.0 / 2.0) * omega / (1 - omega) / N
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        P = np.divide(P, den)
        
        # P1 = np.sum(P, axis=1)
        # Pt1 = np.sum(P, axis=0)
        nu = np.sum(P, axis=1)
        nu_prime = np.sum(P, axis=0)
        N_hat = np.sum(nu_prime)

        # compute X_hat
        nu_tilde = np.kron(nu, np.ones((3,)))
        P_tilde = np.kron(P, np.eye(3))
        X_hat_flat = np.linalg.inv(np.diag(nu_tilde)) @ P_tilde @ X_flat
        # the above array has size (N*3,), and is equivalent to X_hat.flatten(), where X_hat is
        X_hat = np.matmul(np.matmul(np.linalg.inv(np.diag(nu)), P), X)

        # ===== update big_sigma, v_hat, u_hat, and alpha_m_bracket for all m =====
        big_sigma = np.linalg.inv(lam*np.linalg.inv(G) + s**2/sigma2 * np.diag(nu))
        T = np.eye(4)
        T[0:3, 0:3] = s*R
        T[0:3, 3] = t
        T_inv = np.linalg.inv(T)

        X_hat_h = np.hstack((X_hat, np.ones((M, 1))))
        Y_h = np.hstack((Y, np.ones((M, 1))))
        residual = ((T_inv @ X_hat_h.T).T - Y_h)[:, 0:3]
        v_hat = s**2/sigma2 * big_sigma @ np.diag(nu) @ residual  # this is *3 shape
        v_hat_flat = v_hat.flatten()

        u_hat = Y + v_hat
        u_hat_flat = Y_flat + v_hat_flat
        
        alpha_m_bracket = np.exp(scipy.special.digamma(kappa + nu) - scipy.special.digamma(kappa*M + N_hat))
        alpha_m_bracket = np.full((M, N), alpha_m_bracket.reshape(M, 1))

        # ===== update s, R, t, sigma2, y_hat =====
        X_bar = np.sum(np.full((M, 3), nu.reshape(M, 1))*X_hat, axis=0) / N_hat
        sigma2_bar = np.sum(nu*sigma2) / N_hat
        u_bar = np.sum(np.full((M, 3), nu.reshape(M, 1))*u_hat, axis=0) / N_hat

        S_xu = np.zeros((3, 3))
        S_uu = np.zeros((3, 3))
        for m in range (0, M):
            S_xu += nu[m] * (X_hat[m] - X_bar).reshape(3, 1) @ (u_hat[m] - u_bar).reshape(1, 3)
            S_uu += nu[m] * (u_hat[m] - u_bar).reshape(3, 1) @ (u_hat[m] - u_bar).reshape(1, 3)
        S_xu /= N_hat
        S_uu /= N_hat
        S_uu += sigma2_bar*np.eye(3)
        U, _, Vt = np.linalg.svd(S_xu)
        middle_mat = np.eye(3)
        middle_mat[2, 2] = np.linalg.det(U @ Vt.T)
        R = U @ middle_mat @ Vt

        s = np.trace(R @ S_xu) / np.trace(S_uu)
        t = X_bar - s*R @ u_bar

        T_hat = np.eye(4)
        T_hat[0:3, 0:3] = s*R
        T_hat[0:3, 3] = t
        Y_hat = (T_hat @ np.hstack((Y + v_hat, np.ones((M, 1)))).T)[0:3, :].T

        nu_prime_tilde = np.kron(nu_prime, np.ones((3,)))
        sigma2 = 1/(N_hat*3) * (X_flat.reshape(1, N*3) @ np.diag(nu_prime_tilde) @ X_flat.reshape(N*3, 1) - 2*X_flat.reshape(1, N*3) @ P_tilde.T @ Y_hat.flatten() + (Y_hat.flatten()).reshape(1, M*3) @ np.diag(nu_tilde) @ (Y_hat.flatten())) + s**2 * sigma2_bar

        # ===== check convergence =====
        if abs(sigma2 - prev_sigma2) < tol and np.amax(np.abs(Y_hat - prev_Y_hat)) < tol:
            print("Converged after " + str(i) + " iterations. Time taken: " + str(time.time() - start_time) + " s.")
            break

        if i == max_iter - 1:
            print("Optimization did not converge! Time taken: " + str(time.time() - start_time) + " s.")
        
        prev_Y_hat = Y_hat.copy()
        prev_sigma2 = sigma2

    print(s)
    T_hat = np.eye(4)
    T_hat[0:3, 0:3] = R
    T_hat[0:3, 3] = t
    Y_hat = (T_hat @ np.hstack((Y + v_hat, np.ones((M, 1)))).T)[0:3, :].T
    return Y_hat, sigma2

if __name__ == "__main__":
    # load recorded data
    data_dir = dir = join(dirname(abspath(__file__)), "data/frames/")

    # in the BCPD paper, Y are the source nodes
    f = open(data_dir + 'nodes/000_nodes.json', 'rb')
    Y, sigma2 = pkl.load(f, encoding="bytes")
    Y = np.array(Y)
    f.close()

    # ===== load X as target nodes =====
    # f = open(data_dir + '001_nodes.json', 'rb')
    # X, _ = pkl.load(f, encoding="bytes")
    # X = np.array(X)
    # f.close()

    # ===== load X as target point cloud =====
    f = open(data_dir + '001_pcl.json', 'rb')
    X = pkl.load(f, encoding="bytes")
    f.close()
    # downsample X
    X = np.array(X)
    X = X[::int(1/0.025)]

    # run bcpd
    Y_hat, sigma2 = bcpd(X=X, Y=Y, beta=1, omega=0, lam=1, kappa=1e16, gamma=1, max_iter=50, tol=0.00001, sigma2_0=sigma2)

    # test: show both sets of nodes
    Y_pc = Points(Y, c=(255, 0, 0), r=10)
    X_pc = Points(X, c=(0, 0, 255), r=3)
    Y_hat_pc = Points(Y_hat, c=(0, 255, 0), r=10)
    
    plt = Plotter()
    plt.show(Y_pc, X_pc, Y_hat_pc)