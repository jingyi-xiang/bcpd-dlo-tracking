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


def isBetween(x, a, b):
    #check if x is between the segment composed by point a and point b
    if (a <= x <= b or a >= x >= b):
        return True
    else:
        return False


def line_sphere_intersection(point_A, point_B, sphere_center, radius):
    point_A = (point_A[0], point_A[1], point_A[2])
    point_B = (point_B[0], point_B[1], point_B[2])
    sphere_center = (sphere_center[0], sphere_center[1], sphere_center[2])

    # line has two points(pointA and pointB)
    # sphere_center(x, y, z)
    # radius is the sphere's radius
    a = (point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2 + (point_B[2] - point_A[2])**2

    b = 2* ((point_B[0] - point_A[0])*(point_A[0] - sphere_center[0]) + 
            (point_B[1] - point_A[1])*(point_A[1] - sphere_center[1]) + 
            (point_B[2] - point_A[2])*(point_A[2] - sphere_center[2]))
    c = (point_A[0] - sphere_center[0])**2 + (point_A[1] - sphere_center[1])**2 + (point_A[2] - sphere_center[2])**2 - radius**2

    delta = b**2 - (4*a*c)

    d1 = (-b+np.sqrt(delta))/(2*a)
    d2 = (-b-np.sqrt(delta))/(2*a)
    
    if (delta < 0) :
        # print("delta < 0")
        return []
    elif (delta > 0):
        # print("delta > 0")
        # one point
        x1 = point_A[0] + d1*(point_B[0] - point_A[0])
        y1 = point_A[1] + d1*(point_B[1] - point_A[1])
        z1 = point_A[2] + d1*(point_B[2] - point_A[2])
        # the other one
        x2 = point_A[0] + d2*(point_B[0] - point_A[0])
        y2 = point_A[1] + d2*(point_B[1] - point_A[1])
        z2 = point_A[2] + d2*(point_B[2] - point_A[2])

        result = []
        if isBetween((x1, y1, z1), point_A, point_B):
            result.append((x1, y1, z1))
        if isBetween((x2, y2, z2), point_A, point_B):
            result.append((x2, y2, z2))
        if len(result) == 0:
            return []
        else:
            return result
    else:
        # print("delta = 0")
        d1 = -b / (2*a)
        # one point
        x1 = point_A[0] + d1*(point_B[0] - point_A[0])
        y1 = point_A[1] + d1*(point_B[1] - point_A[1])
        z1 = point_A[2] + d1*(point_B[2] - point_A[2])
        if isBetween((x1, y1, z1), point_A, point_B):
            return [(x1, y1, z1)]
        else:
            return []
        
def traverse_euclidean (geodesic_coord, guide_nodes, visible_nodes, alignment, alignment_node_idx=0):
    node_pairs = []

    # extreme cases: only one guide node available
    # since this function will only be called when at least one of head or tail is visible, 
    # the only node will be head or tail
    if len(guide_nodes) == 1:
        node_pairs.append([visible_nodes[0], guide_nodes[0, 0], guide_nodes[0, 1], guide_nodes[0, 2]])
        return np.array(node_pairs)

    if alignment == 0:
        # push back the first pair
        node_pairs.append([visible_nodes[0], guide_nodes[0, 0], guide_nodes[0, 1], guide_nodes[0, 2]])

        consecutive_visible_nodes = []
        for i in range (0, len(visible_nodes)):
            if i == visible_nodes[i]:
                consecutive_visible_nodes.append(i)
            else:
                break
        
        last_found_index = 0
        seg_dist_it = 0
        cur_center = guide_nodes[0]

        # basically pure pursuit
        while (last_found_index+1 <= len(consecutive_visible_nodes)-1) and (seg_dist_it+1 <= len(geodesic_coord)-1):
            look_ahead_dist = np.abs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it])
            found_intersection = False
            intersection = []

            for i in range (last_found_index, len(consecutive_visible_nodes)-1):
                intersections = line_sphere_intersection(guide_nodes[i], guide_nodes[i+1], cur_center, look_ahead_dist)
                intersections = np.array(intersections)

                #  if no intersection found
                if len(intersections) == 0:
                    continue
                elif (len(intersections) == 1) and (pt2pt_dis(intersections[0], guide_nodes[i+1]) > pt2pt_dis(cur_center, guide_nodes[i+1])):
                    continue
                else:
                    found_intersection = True
                    last_found_index = i

                    if len(intersections) == 2:
                        if pt2pt_dis(intersections[0], guide_nodes[i+1]) <= pt2pt_dis(intersections[1], guide_nodes[i+1]):
                            # the first solution is closer
                            intersection = intersections[0].copy()
                            cur_center = intersections[0].copy()
                        else:
                            # the second one is closer
                            intersection = intersections[1].copy()
                            cur_center = intersections[1].copy()
                    else:
                        intersection = intersections[0].copy()
                        cur_center = intersections[0].copy()
                    break
            
            if not found_intersection:
                break
            else:
                node_pairs.append([seg_dist_it+1, intersection[0], intersection[1], intersection[2]])
                seg_dist_it += 1

    elif alignment == 1:
        # push back the first pair
        node_pairs.append([visible_nodes[-1], guide_nodes[-1, 0], guide_nodes[-1, 1], guide_nodes[-1, 2]])

        consecutive_visible_nodes = []
        for i in range (1, len(visible_nodes)+1):
            if visible_nodes[len(visible_nodes)-i] == len(geodesic_coord)-i:
                consecutive_visible_nodes.append(len(geodesic_coord)-i)
            else:
                break
        
        last_found_index = len(guide_nodes) - 1
        seg_dist_it = len(geodesic_coord) - 1
        cur_center = guide_nodes[-1]

        # basically pure pursuit
        while (last_found_index-1 >= len(guide_nodes) - len(consecutive_visible_nodes)) and (seg_dist_it-1 >= 0):
            look_ahead_dist = np.abs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1])
            found_intersection = False
            intersection = []

            for i in range (last_found_index, len(guide_nodes) - len(consecutive_visible_nodes), -1):
                intersections = line_sphere_intersection(guide_nodes[i], guide_nodes[i-1], cur_center, look_ahead_dist)
                intersections = np.array(intersections)

                #  if no intersection found
                if len(intersections) == 0:
                    continue
                elif (len(intersections) == 1) and (pt2pt_dis(intersections[0], guide_nodes[i-1]) > pt2pt_dis(cur_center, guide_nodes[i-1])):
                    continue
                else:
                    found_intersection = True
                    last_found_index = i

                    if len(intersections) == 2:
                        if pt2pt_dis(intersections[0], guide_nodes[i-1]) <= pt2pt_dis(intersections[1], guide_nodes[i-1]):
                            # the first solution is closer
                            intersection = intersections[0].copy()
                            cur_center = intersections[0].copy()
                        else:
                            # the second one is closer
                            intersection = intersections[1].copy()
                            cur_center = intersections[1].copy()
                    else:
                        intersection = intersections[0].copy()
                        cur_center = intersections[0].copy()
                    break
            
            if not found_intersection:
                break
            else:
                node_pairs.append([seg_dist_it-1, intersection[0], intersection[1], intersection[2]])
                seg_dist_it -= 1
    
    else:
        node_pairs.append([visible_nodes[alignment_node_idx], guide_nodes[alignment_node_idx, 0], guide_nodes[alignment_node_idx, 1], guide_nodes[alignment_node_idx, 2]])

        consecutive_visible_nodes_2 = [visible_nodes[alignment_node_idx]]
        for i in range (alignment_node_idx+1, len(visible_nodes)):
            if visible_nodes[i] - visible_nodes[i-1] == 1:
                consecutive_visible_nodes_2.append(visible_nodes[i])
            else:
                break
        
        # ===== traverse from the alignment node to the tail node =====
        last_found_index = alignment_node_idx
        seg_dist_it = visible_nodes[alignment_node_idx]
        cur_center = guide_nodes[alignment_node_idx]

        while (last_found_index+1 <= alignment_node_idx+len(consecutive_visible_nodes_2)-1) and (seg_dist_it+1 <= len(geodesic_coord)-1):
            look_ahead_dist = np.abs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it])
            found_intersection = False
            intersection = []

            for i in range (last_found_index, alignment_node_idx + len(consecutive_visible_nodes_2) - 1):
                intersections = line_sphere_intersection(guide_nodes[i], guide_nodes[i+1], cur_center, look_ahead_dist)
                intersections = np.array(intersections)

                #  if no intersection found
                if len(intersections) == 0:
                    continue
                elif (len(intersections) == 1) and (pt2pt_dis(intersections[0], guide_nodes[i+1]) > pt2pt_dis(cur_center, guide_nodes[i+1])):
                    continue
                else:
                    found_intersection = True
                    last_found_index = i

                    if len(intersections) == 2:
                        if pt2pt_dis(intersections[0], guide_nodes[i+1]) <= pt2pt_dis(intersections[1], guide_nodes[i+1]):
                            # the first solution is closer
                            intersection = intersections[0].copy()
                            cur_center = intersections[0].copy()
                        else:
                            # the second one is closer
                            intersection = intersections[1].copy()
                            cur_center = intersections[1].copy()
                    else:
                        intersection = intersections[0].copy()
                        cur_center = intersections[0].copy()
                    break
            
            if not found_intersection:
                break
            else:
                node_pairs.append([seg_dist_it+1, intersection[0], intersection[1], intersection[2]])
                seg_dist_it += 1
        
        # ===== traverse from the alignment node to the head node =====
        consecutive_visible_nodes_1 = [visible_nodes[alignment_node_idx]]
        for i in range (alignment_node_idx-1, -1, -1):
            if visible_nodes[i+1] - visible_nodes[i] == 1:
                consecutive_visible_nodes_1.append(visible_nodes[i])
            else:
                break
        
        last_found_index = alignment_node_idx
        seg_dist_it = visible_nodes[alignment_node_idx]
        cur_center = guide_nodes[alignment_node_idx]

        while (last_found_index-1 >= alignment_node_idx-len(consecutive_visible_nodes_1)) and (seg_dist_it-1 >= 0):
            look_ahead_dist = np.abs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1])
            found_intersection = False
            intersection = []

            for i in range (last_found_index, 0, -1):
                intersections = line_sphere_intersection(guide_nodes[i], guide_nodes[i-1], cur_center, look_ahead_dist)
                intersections = np.array(intersections)

                #  if no intersection found
                if len(intersections) == 0:
                    continue
                elif (len(intersections) == 1) and (pt2pt_dis(intersections[0], guide_nodes[i-1]) > pt2pt_dis(cur_center, guide_nodes[i-1])):
                    continue
                else:
                    found_intersection = True
                    last_found_index = i

                    if len(intersections) == 2:
                        if pt2pt_dis(intersections[0], guide_nodes[i-1]) <= pt2pt_dis(intersections[1], guide_nodes[i-1]):
                            # the first solution is closer
                            intersection = intersections[0].copy()
                            cur_center = intersections[0].copy()
                        else:
                            # the second one is closer
                            intersection = intersections[1].copy()
                            cur_center = intersections[1].copy()
                    else:
                        intersection = intersections[0].copy()
                        cur_center = intersections[0].copy()
                    break
            
            if not found_intersection:
                break
            else:
                node_pairs.append([seg_dist_it-1, intersection[0], intersection[1], intersection[2]])
                seg_dist_it -= 1
    
    return np.array(node_pairs)

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