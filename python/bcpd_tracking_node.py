#!/usr/bin/env python3

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg

import struct
import time
import cv2
import numpy as np
import scipy

import time
import sys

import message_filters
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from cpd_tracking_node import cpd_lle

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

def register(pts, M, mu=0, max_iter=10):

    # initial guess
    X = pts.copy()
    Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M), np.zeros(M))).T
    if len(pts[0]) == 2:
        Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M))).T
    s = 1
    N = len(pts)
    D = len(pts[0])

    def get_estimates (Y, s):

        # construct the P matrix
        P = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * s) ** (D / 2)
        c = c * mu / (1 - mu)
        c = c * M / N

        P = np.exp(-P / (2 * s))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        P = np.divide(P, den)  # P is M*N
        Pt1 = np.sum(P, axis=0)  # equivalent to summing from 0 to M (results in N terms)
        P1 = np.sum(P, axis=1)  # equivalent to summing from 0 to N (results in M terms)
        Np = np.sum(P1)
        PX = np.matmul(P, X)

        # get new Y
        P1_expanded = np.full((D, M), P1).T
        new_Y = PX / P1_expanded

        # get new sigma2
        Y_N_arr = np.full((N, M, 3), Y)
        Y_N_arr = np.swapaxes(Y_N_arr, 0, 1)
        X_M_arr = np.full((M, N, 3), X)
        diff = Y_N_arr - X_M_arr
        diff = np.square(diff)
        diff = np.sum(diff, 2)
        new_s = np.sum(np.sum(P*diff, axis=1), axis=0) / (Np*D)

        return new_Y, new_s

    prev_Y, prev_s = Y, s
    new_Y, new_s = get_estimates(prev_Y, prev_s)
    # it = 0
    tol = 0.0
    
    for it in range (max_iter):
        prev_Y, prev_s = new_Y, new_s
        new_Y, new_s = get_estimates(prev_Y, prev_s)

    # print(repr(new_x), new_s)
    return new_Y, new_s

def bcpd (X, Y, beta, omega, lam, kappa, gamma, max_iter = 50, tol = 0.00001, sigma2_0=None, dist_transform_values=None, k_vis=None, corr_priors = None, zeta = None):

    # ===== initialization =====
    N = len(X)
    M = len(Y)

    # initialize the J (MxN) matrix (if corr_priors is not None)
    # corr_priors should have format (x, y, z, index)
    # Y_corr = np.hstack((np.arange(25, 35, 1).reshape(len(Y_corr), 1), Y_corr))
    if corr_priors is not None and len(corr_priors) != 0:
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
    seg_dis = np.sqrt(np.sum(np.square(np.diff(Y, axis=0)), axis=1))
    converted_node_coord = []
    last_pt = 0
    converted_node_coord.append(last_pt)
    for i in range (1, M):
        last_pt += seg_dis[i-1]
        converted_node_coord.append(last_pt)
    converted_node_coord = np.array(converted_node_coord)
    converted_node_dis = np.abs(converted_node_coord[None, :] - converted_node_coord[:, None])
    converted_node_dis_sq = np.square(converted_node_dis)
    G = 0.9 * np.exp(-converted_node_dis_sq / (2 * beta**2)) + 0.1 * G
    # G = np.exp(-converted_node_dis / (2 * beta**2))

    # G approximation
    eigen_values, eigen_vectors = np.linalg.eig(G)
    positive_indices = eigen_values > 0
    G_hat = eigen_vectors[:, positive_indices] @ np.diag(eigen_values[positive_indices]) @ eigen_vectors[:, positive_indices].T
    G = G_hat.astype(np.float64)

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

    print("init sigma2 =", sigma2)

    for i in range (0, max_iter):

        # ===== update P and related terms =====
        pts_dis_sq = np.sum((X[None, :, :] - Y_hat[:, None, :]) ** 2, axis=2)
        c = omega / N
        P = alpha_m_bracket * np.exp(-pts_dis_sq / (2 * sigma2)) * (2*np.pi*sigma2)**(-3.0/2.0) * (1-omega) # * np.exp(-s**2 / (2*sigma2) * 3 * np.full((M, N), big_sigma.diagonal().reshape(M, 1))) 
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        P = np.divide(P, den)

        # print(P[:, 0])
        
        # P1 = np.sum(P, axis=1)
        # Pt1 = np.sum(P, axis=0)
        nu = np.sum(P, axis=1)
        nu_prime = np.sum(P, axis=0)
        N_hat = np.sum(nu_prime)

        # compute X_hat
        nu_tilde = np.kron(nu, np.ones(3))
        P_tilde = np.kron(P, np.eye(3))

        try:
            X_hat_flat = (np.linalg.inv(np.diag(nu_tilde)) @ P_tilde @ X_flat).reshape(M*3, 1)
            X_hat = X_hat_flat.reshape(M, 3)
            if np.isnan(X_hat).any():
                print("has nan")
                nu_inv = np.zeros((len(nu),))
                nu_inv[nu > 1e-200] = 1/nu[nu > 1e-200]
                nu_inv_tilde = np.kron(nu_inv, np.ones(3))
                X_hat_flat = (np.diag(nu_inv_tilde) @ P_tilde @ X_flat).reshape(M*3, 1)
                X_hat = X_hat_flat.reshape(M, 3)
        except:
            print("in the except statement")
            nu_inv = np.zeros((len(nu),))
            nu_inv[nu > 1e-200] = 1/nu[nu > 1e-200]
            nu_inv_tilde = np.kron(nu_inv, np.ones(3))
            X_hat_flat = (np.diag(nu_inv_tilde) @ P_tilde @ X_flat).reshape(M*3, 1)
            X_hat = X_hat_flat.reshape(M, 3)

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

        # ===== DO NOT update s, R, t lol =====
        sigma2_bar = np.sum(nu * big_sigma.diagonal()) / N_hat
        
        s = 1
        R = np.eye(3)
        t = np.zeros((3, 1))
        Y_hat = u_hat.copy()

        print("sigma2_bar =", sigma2_bar)
        print("v hat =", v_hat[0])

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
        print("sigma2 =", sigma2)

    return Y_hat, sigma2

def sort_pts(Y_0):
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    N = len(diff)
    G = diff.copy()

    selected_node = np.zeros(N,).tolist()
    selected_node[0] = True
    Y_0_sorted = []
        
    reverse = 0
    counter = 0
    reverse_on = 0
    insertion_counter = 0
    last_visited_b = 0
    while (counter < N - 1):
        
        minimum = 999999
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n

        if len(Y_0_sorted) == 0:
            Y_0_sorted.append(Y_0[a].tolist())
            Y_0_sorted.append(Y_0[b].tolist())
        else:
            if last_visited_b != a:
                reverse += 1
                reverse_on = a
                insertion_counter = 0

            if reverse % 2 == 1:
                # switch direction
                Y_0_sorted.insert(Y_0_sorted.index(Y_0[a].tolist()), Y_0[b].tolist())
            elif reverse != 0:
                Y_0_sorted.insert(Y_0_sorted.index(Y_0[reverse_on].tolist())+1+insertion_counter, Y_0[b].tolist())
                insertion_counter += 1
            else:
                Y_0_sorted.append(Y_0[b].tolist())

        last_visited_b = b
        selected_node[b] = True

        counter += 1

    return np.array(Y_0_sorted)

# original post: https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def ndarray2MarkerArray (Y, node_color, line_color, head):
    results = MarkerArray()
    Y_msg = PointCloud2()
    pc_results_list = []

    for i in range (0, len(Y)):
        cur_node_result = Marker()
        cur_node_result.header = head
        cur_node_result.type = Marker.SPHERE
        cur_node_result.action = Marker.ADD
        cur_node_result.ns = "node_results" + str(i)
        cur_node_result.id = i

        cur_node_result.pose.position.x = Y[i, 0]
        cur_node_result.pose.position.y = Y[i, 1]
        cur_node_result.pose.position.z = Y[i, 2]
        cur_node_result.pose.orientation.w = 1.0
        cur_node_result.pose.orientation.x = 0.0
        cur_node_result.pose.orientation.y = 0.0
        cur_node_result.pose.orientation.z = 0.0

        cur_node_result.scale.x = 0.01
        cur_node_result.scale.y = 0.01
        cur_node_result.scale.z = 0.01
        cur_node_result.color.r = node_color[0]
        cur_node_result.color.g = node_color[1]
        cur_node_result.color.b = node_color[2]
        cur_node_result.color.a = node_color[3]

        results.markers.append(cur_node_result)

        if i == len(Y)-1:
            break

        cur_line_result = Marker()
        cur_line_result.header = head
        cur_line_result.type = Marker.CYLINDER
        cur_line_result.action = Marker.ADD
        cur_line_result.ns = "line_results" + str(i)
        cur_line_result.id = i

        cur_line_result.pose.position.x = ((Y[i] + Y[i+1])/2)[0]
        cur_line_result.pose.position.y = ((Y[i] + Y[i+1])/2)[1]
        cur_line_result.pose.position.z = ((Y[i] + Y[i+1])/2)[2]

        rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), (Y[i+1]-Y[i])/pt2pt_dis(Y[i+1], Y[i])) 
        r = R.from_matrix(rot_matrix)
        x = r.as_quat()[0]
        y = r.as_quat()[1]
        z = r.as_quat()[2]
        w = r.as_quat()[3]

        cur_line_result.pose.orientation.w = w
        cur_line_result.pose.orientation.x = x
        cur_line_result.pose.orientation.y = y
        cur_line_result.pose.orientation.z = z
        cur_line_result.scale.x = 0.005
        cur_line_result.scale.y = 0.005
        cur_line_result.scale.z = pt2pt_dis(Y[i], Y[i+1])
        cur_line_result.color.r = line_color[0]
        cur_line_result.color.g = line_color[1]
        cur_line_result.color.b = line_color[2]
        cur_line_result.color.a = line_color[3]

        results.markers.append(cur_line_result)
        pt = np.array([Y[i,0],Y[i,1],Y[i,2]]).astype(np.float32)
        pc_results_list.append(pt)
    
    # ===== optional publish pc =====
    # pc = np.vstack(pc_results_list).astype(np.float32).T
    # rec_project = np.core.records.fromarrays(pc, 
    #                                          names='x, y, z',
    #                                          formats = 'float32, float32, float32')
    # Y_msg = point_cloud2.array_to_pointcloud2(rec_project, head.stamp, frame_id='camera_color_optical_frame') # include time stamp matching other time
    # track_pc_pub.publish(Y_msg)
    
    return results

def pre_process (X, Y_0, geodesic_coord, total_len, bmask, sigma2_0):

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # log time
    cur_time = time.time()
    guide_nodes, guide_nodes_sigma2_0 = cpd_lle(X, Y_0, 0.5, 1, 1, 0.05, 50, 0.00001, True, True, True, sigma2_0)

    rospy.logwarn('Pre-processing registration: ' + str((time.time() - cur_time)*1000) + ' ms')

    # determine which head node is occluded, if any
    head_visible = False
    tail_visible = False

    if pt2pt_dis(guide_nodes[0], Y_0[0]) < 0.01:
        head_visible = True
    if pt2pt_dis(guide_nodes[-1], Y_0[-1]) < 0.01:
        tail_visible = True

    if not head_visible and not tail_visible:
        rospy.logerr("Neither end visible!")
        # sys.exit() # terminate program
        if pt2pt_dis(guide_nodes[0], Y_0[0]) < pt2pt_dis(guide_nodes[-1], Y_0[-1]):
            head_visible = True
        else:
            tail_visible = True

    cur_total_len = np.sum(np.sqrt(np.sum(np.square(np.diff(guide_nodes, axis=0)), axis=1)))

    # print('tail displacement = ', pt2pt_dis(guide_nodes[-1], Y_0[-1]))
    # print('head displacement = ', pt2pt_dis(guide_nodes[0], Y_0[0]))
    # print('length difference = ', abs(cur_total_len - total_len))

    # visible_dist = np.sum(np.sqrt(np.sum(np.square(np.diff(guide_nodes, axis=0)), axis=1)))
    correspondence_priors = None
    occluded_nodes = None

    mask_dis_threshold = 10
    num_fit_pts = 100
    state = None # 0 for no occlusion, 1 for one tip visible, 2 for two tips visible

    # if abs(cur_total_len - total_len) < 0.02 and head_visible and tail_visible: # (head_visible and tail_visible) or 

    #     rospy.loginfo("Total length unchanged, state = 0")

    #     state = 0
    #     # print('head visible and tail visible or the same len')
    #     correspondence_priors = []
    #     correspondence_priors.append(np.append(np.array([0]), guide_nodes[0]))
    #     correspondence_priors.append(np.append(np.array([len(guide_nodes)-1]), guide_nodes[-1]))

    # log time
    cur_time = time.time()
    
    # elif head_visible and tail_visible:
    if head_visible and tail_visible: # but length condition not met - middle part is occluded
        
        if abs(cur_total_len - total_len) < 0.02:
            rospy.loginfo("Total length unchanged, state = 0")
            return []
            state = 0
        else:
            rospy.loginfo("Both ends visible but total length changed, state = 2")
            state = 2

        # print('head and tail visible but total length changed')

        # first need to determine which portion of the guide nodes are actual useful data (not occupying empty space)
        # determined which nodes are occluded from mask information
        # projection
        guide_nodes_h = np.hstack((guide_nodes, np.ones((len(guide_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, guide_nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)
        # cap uv to be within (1280, 720)
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)
        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))
        # invert bmask for distance transform
        bmask_transformed = scipy.ndimage.distance_transform_edt(255 - bmask)
        # bmask_transformed = bmask_transformed / np.amax(bmask_transformed)
        vis = bmask_transformed[uvs_t]
        valid_guide_nodes_indices = np.where(vis < mask_dis_threshold)[0]

        # # special case: all nodes visible but length is slightly above the threshold
        # if len(valid_guide_nodes_indices) == len(guide_nodes):
        #     correspondence_priors = []
        #     correspondence_priors.append(np.append(np.array([0]), guide_nodes[0]))
        #     correspondence_priors.append(np.append(np.array([len(guide_nodes)-1]), guide_nodes[-1]))
        #     return guide_nodes, np.array(correspondence_priors), None, 0

        # determine a set of nodes for head and a set of nodes for tail
        valid_head_node_indices = []
        for node_idx in range (0, len(guide_nodes)):
            if node_idx in valid_guide_nodes_indices:
                # valid_head_nodes.append(guide_nodes[node_idx])
                valid_head_node_indices.append(node_idx)
            else: 
                break
        if len(valid_head_node_indices) != 0:
            valid_head_nodes = guide_nodes[np.array(valid_head_node_indices)]
        else:
            valid_head_nodes = []
            print('error! no valid head nodes')

        valid_tail_node_indices = []
        for node_idx in range (len(guide_nodes)-1, -1, -1):
            if node_idx in valid_guide_nodes_indices:
                # valid_tail_nodes.append(guide_nodes[node_idx])
                valid_tail_node_indices.append(node_idx)
            else:
                break

        if len(valid_tail_node_indices) != 0:
            valid_tail_nodes = guide_nodes[np.array(valid_tail_node_indices)] # valid tail node is reversed, starting from the end
        else:
            valid_tail_nodes = []
            print('error! no valid tail nodes')

        # initialize a variable for last visible head index and last visible tail index
        last_visible_index_head = None
        last_visible_index_tail = None

        # ----- head visible part -----
        correspondence_priors_head = []

        # if only one head node is visible, should not fit spline
        if len(valid_head_nodes) <= 3:
            correspondence_priors_head = np.hstack((valid_head_node_indices[0], valid_head_nodes[0]))
            last_visible_index_head = 0

        else:
            tck, u = scipy.interpolate.splprep(valid_head_nodes.T, s=0.0001)
            # 1st fit, less points
            u_fine = np.linspace(0, 1, num_fit_pts) # <-- num fit points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

            # 2nd fit, higher accuracy
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1050)
            u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            last_visible_index_head = len(geodesic_coord[geodesic_coord < total_spline_len]) - 1

            # rospy.logwarn("spline total len = " + str(total_spline_len) + "; geodesic_coord = " + str(geodesic_coord[0:last_visible_index_head+1]))

            # geodesic coord is 1D
            correspondence_priors_head = np.vstack((np.arange(0, last_visible_index_head+1), spline_pts[(geodesic_coord[0:last_visible_index_head+1]*1000).astype(int)].T)).T
            # occluded_nodes = np.arange(last_visible_index_head+1, len(Y_0), 1)
        
        # ----- tail visible part -----
        correspondence_priors_tail = []

        # if not enough tail node is visible, should not fit spline
        if len(valid_tail_nodes) <= 3:
            correspondence_priors_tail = np.hstack((valid_tail_node_indices[0], valid_tail_nodes[0]))
            last_visible_index_tail = len(Y_0) - 1

        else:
            tck, u = scipy.interpolate.splprep(valid_tail_nodes.T, s=0.0001)
            # 1st fit, less points
            u_fine = np.linspace(0, 1, num_fit_pts) # <-- num fit points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

            # 2nd fit, higher accuracy
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1050)
            u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            geodesic_coord_from_tail = np.abs(geodesic_coord - geodesic_coord[-1]).tolist()
            geodesic_coord_from_tail.reverse()
            geodesic_coord_from_tail = np.array(geodesic_coord_from_tail)

            last_visible_index_tail = len(Y_0) - len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])

            # rospy.logwarn("spline total len = " + str(total_spline_len) + "; geodesic_coord_from_tail = " + str(geodesic_coord_from_tail[0:len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])]))

            # geodesic coord is 1D
            correspondence_priors_tail = np.vstack((np.arange(len(Y_0)-1, last_visible_index_tail-1, -1), spline_pts[(geodesic_coord_from_tail[0:len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])]*1000).astype(int)].T)).T        
            # occluded_nodes = np.arange(0, last_visible_index_tail, 1)

        # compile occluded nodes
        occluded_nodes = np.arange(last_visible_index_head+1, last_visible_index_tail, 1)
        correspondence_priors = np.vstack((correspondence_priors_head, correspondence_priors_tail))

        # if len(valid_guide_nodes_indices) == len(guide_nodes):
        #     occluded_nodes = None
        #     correspondence_priors = correspondence_priors_head

    elif head_visible and not tail_visible:

        rospy.loginfo("Head visible, state = 1")

        state = 1

        # first need to determine which portion of the guide nodes are actual useful data (not occupying empty space)
        # determined which nodes are occluded from mask information
        # projection
        guide_nodes_h = np.hstack((guide_nodes, np.ones((len(guide_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, guide_nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)
        # temp
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)
        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))
        # invert bmask for distance transform
        bmask_transformed = scipy.ndimage.distance_transform_edt(255 - bmask)
        # bmask_transformed = bmask_transformed / np.amax(bmask_transformed)
        vis = bmask_transformed[uvs_t]
        valid_guide_nodes_indices = np.where(vis < mask_dis_threshold)[0]

        valid_head_node_indices = []
        for node_idx in range (0, len(guide_nodes)):
            if node_idx in valid_guide_nodes_indices:
                # valid_head_nodes.append(guide_nodes[node_idx])
                valid_head_node_indices.append(node_idx)
            else: 
                break
        if len(valid_head_node_indices) != 0:
            valid_head_nodes = guide_nodes[np.array(valid_head_node_indices)]
        else:
            valid_head_nodes = []
            print('error! no valid head nodes')

        correspondence_priors = []

        # if only one head node is visible, should not fit spline
        if len(valid_head_nodes) <= 3:
            correspondence_priors_head = np.hstack((valid_head_node_indices[0], valid_head_nodes[0]))
            last_visible_index_head = 0

        else:
            tck, u = scipy.interpolate.splprep(valid_head_nodes.T, s=0.0001)
            # 1st fit, less points
            u_fine = np.linspace(0, 1, num_fit_pts) # <-- num fit points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

            # 2nd fit, higher accuracy
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1050)
            u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            last_visible_index_head = len(geodesic_coord[geodesic_coord < total_spline_len]) - 1

            # rospy.logwarn("spline total len = " + str(total_spline_len) + "; geodesic_coord = " + str(geodesic_coord[0:last_visible_index_head+1]))

            # geodesic coord is 1D
            correspondence_priors = np.vstack((np.arange(0, last_visible_index_head+1), spline_pts[(geodesic_coord[0:last_visible_index_head+1]*1000).astype(int)].T)).T
        
        occluded_nodes = np.arange(last_visible_index_head+1, len(Y_0), 1)

    elif tail_visible and not head_visible:

        rospy.loginfo("Tail visible, state = 1")

        state = 1

        # first need to determine which portion of the guide nodes are actual useful data (not occupying empty space)
        # determined which nodes are occluded from mask information
        # projection
        guide_nodes_h = np.hstack((guide_nodes, np.ones((len(guide_nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, guide_nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)
        # temp
        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)
        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))
        # invert bmask for distance transform
        bmask_transformed = scipy.ndimage.distance_transform_edt(255 - bmask)
        # bmask_transformed = bmask_transformed / np.amax(bmask_transformed)
        vis = bmask_transformed[uvs_t]
        valid_guide_nodes_indices = np.where(vis < mask_dis_threshold)[0]

        valid_tail_node_indices = []
        for node_idx in range (len(guide_nodes)-1, -1, -1):
            if node_idx in valid_guide_nodes_indices:
                # valid_tail_nodes.append(guide_nodes[node_idx])
                valid_tail_node_indices.append(node_idx)
            else:
                break

        if len(valid_tail_node_indices) != 0:
            valid_tail_nodes = guide_nodes[np.array(valid_tail_node_indices)] # valid tail node is reversed, starting from the end
        else:
            valid_tail_nodes = []
            print('error! no valid tail nodes')

        correspondence_priors = []

        # if only one tail node is visible, should not fit spline
        if len(valid_tail_nodes) <= 3:
            correspondence_priors_tail = np.hstack((valid_tail_node_indices[0], valid_tail_nodes[0]))
            last_visible_index_tail = len(Y_0) - 1

        else:
            tck, u = scipy.interpolate.splprep(valid_tail_nodes.T, s=0.0001)
            # 1st fit, less points
            u_fine = np.linspace(0, 1, num_fit_pts) # <-- num fit points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

            # 2nd fit, higher accuracy
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1050)
            u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
            x_fine, y_fine, z_fine = scipy.interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

            geodesic_coord_from_tail = np.abs(geodesic_coord - geodesic_coord[-1]).tolist()
            geodesic_coord_from_tail.reverse()
            geodesic_coord_from_tail = np.array(geodesic_coord_from_tail)

            last_visible_index_tail = len(Y_0) - len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])

            rospy.logwarn("spline total len = " + str(total_spline_len) + "; geodesic_coord_from_tail = " + str(geodesic_coord_from_tail[0:len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])]))

            # geodesic coord is 1D
            correspondence_priors = np.vstack((np.arange(len(Y_0)-1, last_visible_index_tail-1, -1), spline_pts[(geodesic_coord_from_tail[0:len(geodesic_coord_from_tail[geodesic_coord_from_tail < total_spline_len])]*1000).astype(int)].T)).T        
            
        occluded_nodes = np.arange(0, last_visible_index_tail, 1)
    
    # if none of the above condition is satisfied
    else:
        print('error!')

    rospy.logwarn('Pre-processing fit spline: ' + str((time.time() - cur_time)*1000) + ' ms')

    return np.array(correspondence_priors)

occlusion_mask_rgb = None
def update_occlusion_mask(data):
	global occlusion_mask_rgb
	occlusion_mask_rgb = ros_numpy.numpify(data)

initialized = False
init_nodes = []
nodes = []
cur_time = time.time()
sigma2 = 0
guide_nodes = []
geodesic_coord = []
total_len = 0.0
def callback (rgb, pc):
    global initialized
    global init_nodes
    global nodes
    global cur_time
    global sigma2
    global occlusion_mask_rgb
    global guide_nodes
    global geodesic_coord
    global total_len

    # store header
    head =  std_msgs.msg.Header()
    head.stamp = rgb.header.stamp
    head.frame_id = 'camera_color_optical_frame'

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    # cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    cur_pc = cur_pc.reshape((720, 1280, 3))

    if not use_marker_rope:
        # color thresholding
        lower = (90, 80, 80)
        upper = (130, 255, 255)
        mask = cv2.inRange(hsv_image, lower, upper)
    else:
        # color thresholding
        # --- rope blue ---
        lower = (90, 80, 80)
        upper = (130, 255, 255)
        mask_dlo = cv2.inRange(hsv_image, lower, upper).astype('uint8')

        # --- tape red ---
        lower = (130, 60, 50)
        upper = (255, 255, 255)
        mask_red_1 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
        lower = (0, 60, 50)
        upper = (10, 255, 255)
        mask_red_2 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
        mask_marker = cv2.bitwise_or(mask_red_1.copy(), mask_red_2.copy()).astype('uint8')

        # combine masks
        mask = cv2.bitwise_or(mask_marker.copy(), mask_dlo.copy())

    # process opencv mask
    if occlusion_mask_rgb is None:
        occlusion_mask_rgb = np.ones(cur_image.shape).astype('uint8')*255
    occlusion_mask = cv2.cvtColor(occlusion_mask_rgb.copy(), cv2.COLOR_RGB2GRAY)
    mask = cv2.bitwise_and(mask.copy(), occlusion_mask.copy())

    bmask = mask.copy()
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    # filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.605]

    # temp hard code
    if not use_marker_rope:
        filtered_pc = filtered_pc[filtered_pc[:, 0] > -0.2]
    else:
        filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.58]
        # filtered_pc = filtered_pc[(filtered_pc[:, 2] > 0.58) & (filtered_pc[:, 0] > -0.15) & (filtered_pc[:, 1] > -0.15)]
        # filtered_pc = filtered_pc[~(((filtered_pc[:, 0] < 0.0) & (filtered_pc[:, 1] < 0.05)) | (filtered_pc[:, 2] < 0.58) | (filtered_pc[:, 0] < -0.2) | ((filtered_pc[:, 0] < 0.1) & (filtered_pc[:, 1] < -0.05)))]
    # print('filtered pc shape = ', np.shape(filtered_pc))

    # downsample with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pc)
    downpcd = pcd.voxel_down_sample(voxel_size=0.007)
    filtered_pc = np.asarray(downpcd.points)
    # print('down sampled pc shape = ', np.shape(filtered_pc))

    # add color
    pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
    pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
    filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
    filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

    # filtered_pc = filtered_pc.reshape((len(filtered_pc)*len(filtered_pc[0]), 3))
    header.stamp = rospy.Time.now()
    converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
    pc_pub.publish(converted_points)

    print("Received " + str(len(filtered_pc)) + " points")

    # register nodes
    if not initialized:
        if not use_marker_rope:
            init_nodes, sigma2 = register(filtered_pc, 30, mu=0, max_iter=50)
            init_nodes = np.array(sort_pts(init_nodes))
        else:
            # blob detection
            blob_params = cv2.SimpleBlobDetector_Params()
            blob_params.filterByColor = False
            blob_params.filterByArea = True
            blob_params.filterByCircularity = False
            blob_params.filterByInertia = True
            blob_params.filterByConvexity = False

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(blob_params)
            keypoints_1 = detector.detect(mask_marker)
            keypoints_2 = detector.detect(mask_dlo)

            # Find blob centers in the image coordinates
            blob_image_center = []
            guide_nodes = []

            for i in range(len(keypoints_1)):
                blob_image_center.append((keypoints_1[i].pt[0],keypoints_1[i].pt[1]))
                cur_pt = cur_pc[int(keypoints_1[i].pt[1]), int(keypoints_1[i].pt[0])]
                if cur_pt[2] > 0.55:
                    guide_nodes.append(cur_pt)

            for i in range(len(keypoints_2)):
                blob_image_center.append((keypoints_2[i].pt[0],keypoints_2[i].pt[1]))
                cur_pt = cur_pc[int(keypoints_2[i].pt[1]), int(keypoints_2[i].pt[0])]
                if cur_pt[2] > 0.55:
                    guide_nodes.append(cur_pt)

            sigma2 = 1e-5
            init_nodes = np.array(sort_pts(np.array(guide_nodes)))
        
        # compute preset coord and total len. one time action
        seg_dis = np.sqrt(np.sum(np.square(np.diff(init_nodes, axis=0)), axis=1))
        geodesic_coord = []
        last_pt = 0
        geodesic_coord.append(last_pt)
        for i in range (1, len(init_nodes)):
            last_pt += seg_dis[i-1]
            geodesic_coord.append(last_pt)
        geodesic_coord = np.array(geodesic_coord)
        total_len = np.sum(np.sqrt(np.sum(np.square(np.diff(init_nodes, axis=0)), axis=1)))

        nodes = init_nodes
        initialized = True
        print("sigma2 =", sigma2)
        print("Initialized")
    else:
        # determined which nodes are occluded from mask information
        mask_dis_threshold = 10
        # projection
        init_nodes_h = np.hstack((init_nodes, np.ones((len(init_nodes), 1))))
        image_coords = np.matmul(proj_matrix, init_nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        us = np.where(us >= 1280, 1279, us)
        vs = np.where(vs >= 720, 719, vs)

        uvs = np.vstack((vs, us)).T
        uvs_t = tuple(map(tuple, uvs.T))

        # invert bmask for distance transform
        bmask_transformed = scipy.ndimage.distance_transform_edt(255 - bmask)
        # bmask_transformed = bmask_transformed / np.amax(bmask_transformed)
        vis = bmask_transformed[uvs_t]

        corr_priors = pre_process(filtered_pc, nodes, geodesic_coord, total_len, bmask, sigma2)

        # ===== Parameters =====
        # X \in R^N  -- target point set
        # Y \in R^M  -- source point set 
        # omega      -- the outlier probability
        # kappa      -- the parameter of the Dirichlet distribution used as a prior distribution of alpha
        # gamma      -- the scale factor of sigma2_0
        # beta       -- controls the influence of motion coherence
        # filtered_pc *= 100
        # nodes *= 100
        # if len(corr_priors) != 0:
        #     corr_priors[:, 1:4] *= 100
        nodes, sigma2 = bcpd(X=filtered_pc, Y=nodes, beta=10, omega=0.0, lam=1, kappa=1e16, gamma=1, max_iter=50, tol=0.0001, sigma2_0=sigma2, dist_transform_values=None, k_vis=0.3, corr_priors=corr_priors, zeta=1e-3)
        # print("sigma2 =", sigma2)
        # filtered_pc /= 100
        # nodes /= 100
        # if len(corr_priors) != 0:
        #     corr_priors[:, 1:4] /= 100
        init_nodes = nodes.copy()

        # project and pub tracking image
        nodes_h = np.hstack((nodes, np.ones((len(nodes), 1))))

        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        cur_image_masked = cv2.bitwise_and(cur_image, occlusion_mask_rgb)
        tracking_img = (cur_image*0.5 + cur_image_masked*0.5).astype(np.uint8)

        for i in range (len(image_coords)):
            uv = (us[i], vs[i])

            # draw line
            if i != len(image_coords)-1:
                if vis[i] < mask_dis_threshold:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (0, 255, 0), 2)
                else:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (255, 0, 0), 2)
            
            # draw circle
            if vis[i] < mask_dis_threshold:
                cv2.circle(tracking_img, uv, 5, (255, 150, 0), -1)
            else:
                cv2.circle(tracking_img, uv, 5, (255, 0, 0), -1)
        
        tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
        tracking_img_msg.header = head
        tracking_img_pub.publish(tracking_img_msg)

    results = ndarray2MarkerArray(nodes, [255, 150, 0, 0.75], [0, 255, 0, 0.75], head)
    results_pub.publish(results)

    print("Callback total:", time.time() - cur_time)
    cur_time = time.time()

use_marker_rope = True

if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)
    # depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)
    occlusion_sub = rospy.Subscriber('/mask_with_occlusion', Image, update_occlusion_mask)

    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)
    results_pub = rospy.Publisher ('/results', MarkerArray, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()