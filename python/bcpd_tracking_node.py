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
    # G = np.exp(-converted_node_dis_sq / (2 * beta**2))

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

occlusion_mask_rgb = None
def update_occlusion_mask(data):
	global occlusion_mask_rgb
	occlusion_mask_rgb = ros_numpy.numpify(data)

initialized = False
init_nodes = []
nodes = []
cur_time = time.time()
sigma2 = 0
def callback (rgb, pc):
    global initialized
    global init_nodes
    global nodes
    global cur_time
    global sigma2
    global occlusion_mask_rgb

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

    # --- rope blue ---
    lower = (90, 60, 40)
    upper = (130, 255, 255)
    mask = cv2.inRange(hsv_image, lower, upper)
    bmask = mask.copy()
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    filtered_pc = cur_pc*mask
    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
    # filtered_pc = filtered_pc[filtered_pc[:, 2] < 0.605]
    filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.4]
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

    # process opencv mask
    if occlusion_mask_rgb is None:
        occlusion_mask_rgb = np.ones(cur_image.shape).astype('uint8')*255
    occlusion_mask = cv2.cvtColor(occlusion_mask_rgb.copy(), cv2.COLOR_RGB2GRAY)

    # register nodes
    if not initialized:
        # get nodes for wire 3
        init_nodes, sigma2 = register(filtered_pc, 30, mu=0, max_iter=50)
        init_nodes = np.array(sort_pts(init_nodes))
        initialized = True
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

        # ===== Parameters =====
        # X \in R^N  -- target point set
        # Y \in R^M  -- source point set 
        # omega      -- the outlier probability
        # kappa      -- the parameter of the Dirichlet distribution used as a prior distribution of alpha
        # gamma      -- the scale factor of sigma2_0
        # beta       -- controls the influence of motion coherence
        nodes, sigma2 = bcpd(X=filtered_pc, Y=init_nodes, beta=0.1, omega=0.05, lam=10, kappa=1e16, gamma=10, max_iter=50, tol=0.00001, sigma2_0=sigma2)
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
            # draw circle
            uv = (us[i], vs[i])
            if vis[i] < mask_dis_threshold:
                cv2.circle(tracking_img, uv, 5, (255, 150, 0), -1)
            else:
                cv2.circle(tracking_img, uv, 5, (255, 0, 0), -1)

            # draw line
            if i != len(image_coords)-1:
                if vis[i] < mask_dis_threshold:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (0, 255, 0), 2)
                else:
                    cv2.line(tracking_img, uv, (us[i+1], vs[i+1]), (255, 0, 0), 2)
        
        tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
        tracking_img_msg.header = head
        tracking_img_pub.publish(tracking_img_msg)

        results = ndarray2MarkerArray(nodes, [255, 150, 0, 0.75], [0, 255, 0, 0.75], head)
        results_pub.publish(results)

        print(time.time() - cur_time)
        cur_time = time.time()

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

    tracking_img_pub = rospy.Publisher ('/tracking_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)
    results_pub = rospy.Publisher ('/results', MarkerArray, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()