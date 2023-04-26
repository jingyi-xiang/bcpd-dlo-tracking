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

def bcpd (X, Y, beta, omega, lam, kappa, gamma, max_iter = 50, tol = 0.00001, sigma2_0 = None, corr_priors = None, zeta = None):
    # ===== initialization =====
    N = len(X)
    M = len(Y)

    # initialize the J (MxN) matrix (if corr_priors is not None)
    # corr_priors should have format (x, y, z, index)
    # Y_corr = np.hstack((np.arange(25, 35, 1).reshape(len(Y_corr), 1), Y_corr))
    if corr_priors is not None:
        N += len(corr_priors)
        J = np.zeros((M, N))
        X = np.vstack((corr_priors[:, 1:4], X))
        for i in range (0, len(corr_priors)):
            J[int(corr_priors[i, 0]), i] = 1

    X_flat = X.flatten().reshape(N*3, 1)
    Y_flat = Y.flatten().reshape(M*3, 1)

    Y_hat = Y.copy()
    v_hat = np.zeros((M, 3))
    Y_hat_flat = Y_hat.flatten().reshape(M*3, 1)
    v_hat_flat = v_hat.flatten().reshape(M*3, 1)

    big_sigma = np.eye(M)
    alpha_m_bracket = np.ones((M, N)) * 1.0/M
    s = 1
    R = np.eye(3)
    t = np.zeros((3, 1))

    # initialize G
    diff = Y[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    G = np.exp(-diff / (2 * beta**2))

    # geodesic distance
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
    # # G = np.exp(-converted_node_dis / (2 * beta**2))

    # # G approximation
    # eigen_values, eigen_vectors = np.linalg.eig(G)
    # positive_indices = eigen_values > 0
    # G_hat = eigen_vectors[:, positive_indices] @ np.diag(eigen_values[positive_indices]) @ eigen_vectors[:, positive_indices].T
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

        # ===== update P and related terms =====
        pts_dis_sq = np.sum((X[None, :, :] - Y_hat[:, None, :]) ** 2, axis=2)
        c = omega / N
        P = alpha_m_bracket * np.exp(-pts_dis_sq / (2 * sigma2)) * np.exp(-s**2 / (2*sigma2) * 3 * np.full((M, N), big_sigma.diagonal().reshape(M, 1))) * (2*np.pi*sigma2)**(-3.0/2.0) * (1-omega)
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
        nu_tilde = np.kron(nu, np.ones(3))
        P_tilde = np.kron(P, np.eye(3))
        X_hat_flat = (np.linalg.inv(np.diag(nu_tilde)) @ P_tilde @ X_flat).reshape(M*3, 1)
        X_hat = X_hat_flat.reshape(M, 3)

        # try:
        #     X_hat = np.linalg.inv(np.diag(nu)) @ P @ X
        #     if np.isnan(X_hat).any():
        #         nu_inv = np.zeros((len(nu),))
        #         nu_inv[nu > 1/8**257] = 1/nu[nu > 1/8**257]
        #         X_hat = np.diag(nu_inv) @ P @ X
        # except:
        #     nu_inv = np.zeros((len(nu),))
        #     nu_inv[nu > 1/8**257] = 1/nu[nu > 1/8**257]
        #     X_hat = np.diag(nu_inv) @ P @ X

        # ===== update big_sigma, v_hat, u_hat, and alpha_m_bracket for all m =====\
        if corr_priors is None or len(corr_priors) == 0:
            big_sigma = np.linalg.inv(lam*np.linalg.inv(G) + s**2/sigma2 * np.diag(nu))
            big_sigma_tilde = np.kron(big_sigma, np.eye(3))
            R_tilde = np.kron(np.eye(M), R)
            t_tilde = np.kron(np.ones((M, 1)), t)

            residual = 1/s * R_tilde.T @ (X_hat_flat - t_tilde) - Y_flat
            v_hat_flat = s**2 / sigma2 * big_sigma_tilde @ np.diag(nu_tilde) @ residual
            v_hat = v_hat_flat.reshape(M, 3)
        else:
            # create variables for corr_priors
            nu_corr = np.sum(J, axis=1)
            nu_corr_prime = np.sum(J, axis=0)
            nu_corr_tilde = np.kron(nu_corr, np.ones(3))
            J_tilde = np.kron(J, np.eye(3))

            big_sigma = np.linalg.inv(lam*np.linalg.inv(G) + s**2/sigma2 * np.diag(nu) + s**2/zeta * np.diag(nu_corr))
            big_sigma_tilde = np.kron(big_sigma, np.eye(3))
            R_tilde = np.kron(np.eye(M), R)
            t_tilde = np.kron(np.ones((M, 1)), t)

            residual = 1/s * R_tilde.T @ (X_hat_flat - t_tilde) - Y_flat
            dv_residual = np.diag(nu_corr_tilde) @ (1/s*R_tilde.T @ J_tilde @ X_flat - 1/s*R_tilde.T @ t_tilde - Y_flat)

            v_hat_flat = s**2 / sigma2 * big_sigma_tilde @ np.diag(nu_tilde) @ residual + s**2 / zeta * big_sigma_tilde @ dv_residual
            v_hat = v_hat_flat.reshape(M, 3)

        u_hat_flat = Y_flat + v_hat_flat
        u_hat = u_hat_flat.reshape(M, 3)
        
        alpha_m_bracket = np.exp(scipy.special.digamma(kappa + nu) - scipy.special.digamma(kappa*M + N_hat))
        alpha_m_bracket = np.full((M, N), alpha_m_bracket.reshape(M, 1))

        # ===== update s, R, t, sigma2, y_hat =====
        X_bar = np.sum(np.full((M, 3), nu.reshape(M, 1))*X_hat, axis=0) / N_hat
        u_bar = np.sum(np.full((M, 3), nu.reshape(M, 1))*u_hat, axis=0) / N_hat

        S_xu = np.zeros((3, 3))
        S_uu = np.zeros((3, 3))
        for m in range (0, M):
            S_xu += nu[m] * (X_hat[m] - X_bar).reshape(3, 1) @ (u_hat[m] - u_bar).reshape(1, 3)
            S_uu += nu[m] * (u_hat[m] - u_bar).reshape(3, 1) @ (u_hat[m] - u_bar).reshape(1, 3)
        S_xu /= N_hat
        S_uu /= N_hat

        sigma2_bar = np.sum(nu * big_sigma.diagonal()) / N_hat
        S_uu += sigma2_bar*np.eye(3)
        U, _, Vt = np.linalg.svd(S_xu)
        middle_mat = np.eye(3)
        middle_mat[2, 2] = np.linalg.det(U @ Vt.T)
        R = U @ middle_mat @ Vt

        s = np.trace(R @ S_xu) / np.trace(S_uu)
        t = (X_bar - s*R @ u_bar).reshape(3, 1)

        T_hat = np.eye(4)
        T_hat[0:3, 0:3] = s*R
        T_hat[0:3, 3] = t.reshape(3,)
        Y_hat = (T_hat @ np.hstack((Y + v_hat, np.ones((M, 1)))).T)[0:3, :].T

        nu_prime_tilde = np.kron(nu_prime, np.ones(3))
        sigma2 = 1/(N_hat*3) * (X_flat.reshape(1, N*3) @ np.diag(nu_prime_tilde) @ X_flat.reshape(N*3, 1) - 2*X_flat.reshape(1, N*3) @ P_tilde.T @ Y_hat.flatten() + (Y_hat.flatten()).reshape(1, M*3) @ np.diag(nu_tilde) @ (Y_hat.flatten())) + s**2 * sigma2_bar
        sigma2 = sigma2[0, 0]

        # ===== check convergence =====
        if abs(sigma2 - prev_sigma2) < tol and np.amax(np.abs(Y_hat - prev_Y_hat)) < tol:
            print("Converged after " + str(i) + " iterations. Time taken: " + str(time.time() - start_time) + " s.")
            break

        if i == max_iter - 1:
            print("Optimization did not converge! Time taken: " + str(time.time() - start_time) + " s.")
        
        prev_Y_hat = Y_hat.copy()
        prev_sigma2 = sigma2

        # # test: show both sets of nodes
        # Y_pc = Points(Y, c=(255, 0, 0), alpha=0.5, r=20)
        # X_pc = Points(X, c=(0, 0, 0), r=8)
        # Y_hat_pc = Points(Y_hat, c=(0, 255, 0), alpha=0.5, r=20)
        
        # plt = Plotter()
        # plt.show(Y_pc, X_pc, Y_hat_pc)
        # print(sigma2)

    return Y_hat, sigma2

if __name__ == "__main__":
    # load recorded data
    data_dir = dir = join(dirname(abspath(__file__)), "data/frames/")

    # in the BCPD paper, Y are the source nodes
    f = open(data_dir + 'nodes/000_nodes.json', 'rb')
    Y, sigma2 = pkl.load(f, encoding="bytes")
    Y = np.array(Y)
    f.close()

    # ===== load correspondence priors =====
    f = open(data_dir + 'nodes/001_nodes.json', 'rb')
    Y_corr, _ = pkl.load(f, encoding="bytes")
    f.close()
    Y_corr = np.flip(Y_corr, 0)
    Y_corr = np.array(Y_corr)[25:35, :]
    Y_corr = np.hstack((np.arange(25, 35, 1).reshape(len(Y_corr), 1), Y_corr))

    # ===== load X as target point cloud =====
    f = open(data_dir + '001_pcl.json', 'rb')
    X = pkl.load(f, encoding="bytes")
    f.close()
    # downsample X
    X = np.array(X)
    X = X[::int(1/0.025)]

    # occlusion
    X = X[X[:, 0] > -0.05]

    # run bcpd
    Y_hat, sigma2 = bcpd(X=X, Y=Y, beta=2, omega=0.0, lam=1, kappa=1e16, gamma=1, max_iter=100, tol=0.0001, sigma2_0=None, corr_priors=Y_corr, zeta=1e-4)

    # test: show both sets of nodes
    Y_pc = Points(Y, c=(255, 0, 0), alpha=0.5, r=20)
    X_pc = Points(X, c=(0, 0, 0), r=8)
    Y_hat_pc = Points(Y_hat, c=(0, 255, 0), alpha=0.5, r=20)
    Y_corr_pc = Points(Y_corr[:, 1:4], c=(0, 0, 255), alpha=0.5, r=20)
    
    plt = Plotter()
    plt.show(Y_pc, X_pc, Y_hat_pc, Y_corr_pc)

    # more frames?
    num_of_frames = 0
    for i in range (2, num_of_frames):  # the next frame is frame 2
        sample_prefix = ''
        if len(str(i)) == 1:
            sample_prefix = '00'
        elif len(str(i)) == 2:
            sample_prefix ='0'
        else:
            sample_prefix = ''
        sample_id = sample_prefix + str(i)
        # sample_id = '001'

        f = open(data_dir + sample_id + '_pcl.json', 'rb')
        X = pkl.load(f, encoding="bytes")
        f.close()

        X = np.array(X)
        X = X[::int(1/0.05)]

        # run bcpd
        Y_hat, sigma2 = bcpd(X=X, Y=Y_hat, beta=0.01, omega=0.0, lam=1, kappa=1e16, gamma=1, max_iter=50, tol=0.0001, sigma2_0=None)

        # test: show both sets of nodes
        Y_pc = Points(Y, c=(255, 0, 0), r=10)
        X_pc = Points(X, c=(0, 0, 255), r=3)
        Y_hat_pc = Points(Y_hat, c=(0, 255, 0), r=10)
        
        plt = Plotter()
        plt.show(Y_pc, X_pc, Y_hat_pc)