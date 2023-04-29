#include "../include/utils.h"
#include "../include/bcpd_tracker.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using cv::Mat;

bcpd_tracker::bcpd_tracker () {}

bcpd_tracker::bcpd_tracker(int num_of_nodes) 
{
    // default initialize
    Y_ = MatrixXd::Zero(num_of_nodes, 3);
    sigma2_ = 0.0;
    beta_ = 0.1;
    omega_ = 0.05;
    lambda_ = 10;
    kappa_ = 1e16;
    gamma_ = 10;
    zeta_ = 1e-4;
    max_iter_ = 50;
    tol_ = 0.00001;
    use_prev_sigma2_ = false;
}

bcpd_tracker::bcpd_tracker(int num_of_nodes,
                           double beta,
                           double lambda,
                           double omega,
                           double kappa,
                           double gamma,
                           double zeta,
                           int max_iter,
                           const double tol,
                           bool use_prev_sigma2)
{
    Y_ = MatrixXd::Zero(num_of_nodes, 3);
    sigma2_ = 0.0;
    beta_ = beta;
    omega_ = omega;
    lambda_ = lambda;
    kappa_ = kappa;
    gamma_ = gamma;
    zeta_ = zeta;
    max_iter_ = max_iter;
    tol_ = tol;
    use_prev_sigma2_ = use_prev_sigma2;
}

double bcpd_tracker::get_sigma2 () {
    return sigma2_;
}

MatrixXd bcpd_tracker::get_tracking_result () {
    return Y_;
}

MatrixXd bcpd_tracker::get_guide_nodes () {
    return guide_nodes_;
}

std::vector<MatrixXd> bcpd_tracker::get_correspondence_pairs () {
    return correspondence_priors_;
}

void bcpd_tracker::initialize_nodes (MatrixXd Y_init) {
    Y_ = Y_init.replicate(1, 1);
}

void bcpd_tracker::initialize_geodesic_coord (std::vector<double> geodesic_coord) {
    for (int i = 0; i < geodesic_coord.size(); i ++) {
        geodesic_coord_.push_back(geodesic_coord[i]);
    }
}

void bcpd_tracker::set_sigma2 (double sigma2) {
    sigma2_ = sigma2;
}

std::vector<int> bcpd_tracker::get_nearest_indices (int k, int M, int idx) {
    std::vector<int> indices_arr;
    if (idx - k < 0) {
        for (int i = 0; i <= idx + k; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }
    else if (idx + k >= M) {
        for (int i = idx - k; i <= M - 1; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }
    else {
        for (int i = idx - k; i <= idx + k; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }

    return indices_arr;
}

MatrixXd bcpd_tracker::calc_LLE_weights (int k, MatrixXd X) {
    MatrixXd W = MatrixXd::Zero(X.rows(), X.rows());
    for (int i = 0; i < X.rows(); i ++) {
        std::vector<int> indices = get_nearest_indices(static_cast<int>(k/2), X.rows(), i);
        MatrixXd xi = X.row(i);
        MatrixXd Xi = MatrixXd(indices.size(), X.cols());

        // fill in Xi: Xi = X[indices, :]
        for (int r = 0; r < indices.size(); r ++) {
            Xi.row(r) = X.row(indices[r]);
        }

        // component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        MatrixXd component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
        MatrixXd Gi = component.transpose() * component;
        MatrixXd Gi_inv;

        if (Gi.determinant() != 0) {
            Gi_inv = Gi.inverse();
        }
        else {
            // std::cout << "Gi singular at entry " << i << std::endl;
            double epsilon = 0.00001;
            Gi.diagonal().array() += epsilon;
            Gi_inv = Gi.inverse();
        }

        // wi = Gi_inv * 1 / (1^T * Gi_inv * 1)
        MatrixXd ones_row_vec = MatrixXd::Constant(1, Xi.rows(), 1.0);
        MatrixXd ones_col_vec = MatrixXd::Constant(Xi.rows(), 1, 1.0);

        MatrixXd wi = (Gi_inv * ones_col_vec) / (ones_row_vec * Gi_inv * ones_col_vec).value();
        MatrixXd wi_T = wi.transpose();

        for (int c = 0; c < indices.size(); c ++) {
            W(i, indices[c]) = wi_T(c);
        }
    }

    return W;
}

void bcpd_tracker::cpd_lle (MatrixXd X,
                            MatrixXd& Y,
                            double& sigma2,
                            double beta,
                            double lambda,
                            double gamma,
                            double mu,
                            int max_iter,
                            double tol,
                            bool include_lle,
                            bool use_geodesic,
                            bool use_prev_sigma2)
{
    int M = Y.rows();
    int N = X.rows();
    int D = 3;

    // initialization
    // compute differences for G matrix computation
    MatrixXd diff_yy = MatrixXd::Zero(M, M);
    MatrixXd diff_yy_sqrt = MatrixXd::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y.row(i) - Y.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y.row(i) - Y.row(j)).norm();
        }
    }

    MatrixXd G = (-diff_yy / (2 * beta * beta)).array().exp();
    MatrixXd Y_0 = Y.replicate(1, 1);

    // diff_xy should be a (M * N) matrix
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    if (!use_prev_sigma2 || sigma2 == 0) {
        sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);
    }

    for (int it = 0; it < max_iter; it ++) {
        // ----- E step: compute posteriori probability matrix P -----

        // update diff_xy
        diff_xy = MatrixXd::Zero(M, N);
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        double c = std::pow(2 * M_PI * sigma2, static_cast<double>(D) / 2);
        c *= mu / (1 - mu);
        c *= static_cast<double>(M) / N;

        MatrixXd P = (-diff_xy / (2 * sigma2)).array().exp().matrix();

        RowVectorXd den = P.colwise().sum();
        den.array() += c;

        P = P.array().rowwise() / den.array();

        MatrixXd Pt1 = P.colwise().sum();  // this should have shape (N,) or (1, N)
        MatrixXd P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXd PX = P * X;

        // M step
        MatrixXd A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M);
        MatrixXd B_matrix = PX - P1.asDiagonal() * Y_0;

        // MatrixXd W = A_matrix.householderQr().solve(B_matrix);
        // MatrixXd W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);
        MatrixXd W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);

        MatrixXd T = Y_0 + G * W;
        double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        double trPXtT = (PX.transpose() * T).trace();
        double trTtdP1T = (T.transpose() * P1.asDiagonal() * T).trace();

        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);

        if (pt2pt_dis_sq(Y, Y_0 + G*W) < tol) {
            Y = Y_0 + G*W;
            ROS_INFO_STREAM("Iteration until convergence: " + std::to_string(it+1));
            break;
        }
        else {
            Y = Y_0 + G*W;
        }

        if (it == max_iter - 1) {
            ROS_ERROR("optimization did not converge!");
            break;
        }
    }
}

// alignment: 0 --> align with head; 1 --> align with tail
std::vector<MatrixXd> bcpd_tracker::traverse_geodesic (std::vector<double> geodesic_coord, const MatrixXd guide_nodes, const std::vector<int> visible_nodes, int alignment) {
    std::vector<MatrixXd> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail is visible, 
    // the only node will be head or tail
    if (guide_nodes.rows() == 1) {
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    double guide_nodes_total_dist = 0;
    double total_seg_dist = 0;
    
    if (alignment == 0) {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        // initialize iterators
        int guide_nodes_it = 0;
        int seg_dist_it = 0;
        int last_seg_dist_it = seg_dist_it;

        // ultimate terminating condition: run out of guide nodes to use. two conditions that can trigger this:
        //   1. next visible node index - current visible node index > 1
        //   2. currenting using the last two guide nodes
        while (visible_nodes[guide_nodes_it+1] - visible_nodes[guide_nodes_it] == 1 && guide_nodes_it+1 <= guide_nodes.rows()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1));
            // now keep adding segment dists until the total seg dists exceed the current total guide node dists
            while (guide_nodes_total_dist > total_seg_dist) {
                // break condition
                if (seg_dist_it == geodesic_coord.size()-1) {
                    break;
                }

                total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it+1]);
                if (total_seg_dist <= guide_nodes_total_dist) {
                    seg_dist_it += 1;
                }
                else {
                    total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it+1]);
                    break;
                }
            }
            // additional break condition
            if (seg_dist_it == geodesic_coord.size()-1) {
                break;
            }
            // upon exit, seg_dist_it will be at the locaiton where the total seg dist is barely smaller than guide nodes total dist
            // the node desired should be in between guide_nodes[guide_nodes_it] and guide_node[guide_nodes_it + 1]
            // seg_dist_it will also be within guide_nodes_it and guide_nodes_it + 1
            if (guide_nodes_it == 0 && seg_dist_it == 0) {
                continue;
            }
            // if one guide nodes segment is not long enough
            if (last_seg_dist_it == seg_dist_it) {
                guide_nodes_it += 1;
                continue;
            }
            double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1)));
            MatrixXd temp = (guide_nodes.row(guide_nodes_it + 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1));
            node_pair(0, 0) = seg_dist_it;
            node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
            node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
            node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
            node_pairs.push_back(node_pair);

            // update guide_nodes_it at the very end
            guide_nodes_it += 1;
            last_seg_dist_it = seg_dist_it;
        }
    }
    else {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        // initialize iterators
        int guide_nodes_it = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        int last_seg_dist_it = seg_dist_it;

        // ultimate terminating condition: run out of guide nodes to use. two conditions that can trigger this:
        //   1. next visible node index - current visible node index > 1
        //   2. currenting using the last two guide nodes
        while (visible_nodes[guide_nodes_it] - visible_nodes[guide_nodes_it-1] == 1 && guide_nodes_it-1 >= 0 && seg_dist_it-1 >= 0) {
            guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1));
            // now keep adding segment dists until the total seg dists exceed the current total guide node dists
            while (guide_nodes_total_dist > total_seg_dist) {
                // break condition
                if (seg_dist_it == 0) {
                    break;
                }

                total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
                if (total_seg_dist <= guide_nodes_total_dist) {
                    seg_dist_it -= 1;
                }
                else {
                    total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
                    break;
                }
            }
            // additional break condition
            if (seg_dist_it == 0) {
                break;
            }
            // upon exit, seg_dist_it will be at the locaiton where the total seg dist is barely smaller than guide nodes total dist
            // the node desired should be in between guide_nodes[guide_nodes_it] and guide_node[guide_nodes_it + 1]
            // seg_dist_it will also be within guide_nodes_it and guide_nodes_it + 1
            if (guide_nodes_it == 0 && seg_dist_it == 0) {
                continue;
            }
            // if one guide nodes segment is not long enough
            if (last_seg_dist_it == seg_dist_it) {
                guide_nodes_it -= 1;
                continue;
            }
            double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1)));
            MatrixXd temp = (guide_nodes.row(guide_nodes_it - 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1));
            node_pair(0, 0) = seg_dist_it;
            node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
            node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
            node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
            node_pairs.insert(node_pairs.begin(), node_pair);

            // update guide_nodes_it at the very end
            guide_nodes_it -= 1;
            last_seg_dist_it = seg_dist_it;
        }
    }

    return node_pairs;
}

// overload
std::vector<MatrixXd> bcpd_tracker::traverse_euclidean (std::vector<double> geodesic_coord, const MatrixXd guide_nodes, const std::vector<int> visible_nodes, int alignment, int alignment_node_idx) {
    std::vector<MatrixXd> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail is visible, 
    // the only node will be head or tail
    if (guide_nodes.rows() == 1) {
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    if (alignment == 0) {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 0; i < visible_nodes.size(); i ++) {
            if (i == visible_nodes[i]) {
                consecutive_visible_nodes.push_back(i);
            }
            else {
                break;
            }
        }

        int last_found_index = 0;
        int seg_dist_it = 0;
        MatrixXd cur_center = guide_nodes.row(0);

        // basically pure pursuit lol
        while (last_found_index+1 <= consecutive_visible_nodes.size()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= consecutive_visible_nodes.size()-1; i ++) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it + 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it += 1;
            }
        }
    }
    else if (alignment == 1){
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 1; i <= visible_nodes.size(); i ++) {
            if (visible_nodes[visible_nodes.size()-i] == geodesic_coord.size()-i) {
                consecutive_visible_nodes.push_back(geodesic_coord.size()-i);
            }
            else {
                break;
            }
        }

        int last_found_index = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        MatrixXd cur_center = guide_nodes.row(guide_nodes.rows()-1);

        // basically pure pursuit lol
        while (last_found_index-1 >= (guide_nodes.rows() - consecutive_visible_nodes.size()) && seg_dist_it-1 >= 0) {

            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);

            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i >= (guide_nodes.rows() - consecutive_visible_nodes.size() + 1); i --) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it - 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it -= 1;
            }
        }
    }
    else {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[alignment_node_idx], guide_nodes(alignment_node_idx, 0), guide_nodes(alignment_node_idx, 1), guide_nodes(alignment_node_idx, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes_2 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx+1; i < visible_nodes.size(); i ++) {
            if (visible_nodes[i] - visible_nodes[i-1] == 1) {
                consecutive_visible_nodes_2.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        // ----- traverse from the alignment node to the tail node -----
        int last_found_index = alignment_node_idx;
        int seg_dist_it = visible_nodes[alignment_node_idx];
        MatrixXd cur_center = guide_nodes.row(alignment_node_idx);

        // basically pure pursuit lol
        while (last_found_index+1 <= alignment_node_idx+consecutive_visible_nodes_2.size()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= alignment_node_idx+consecutive_visible_nodes_2.size()-1; i ++) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it + 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it += 1;
            }
        }


        // ----- traverse from alignment node to head node -----
        std::vector<int> consecutive_visible_nodes_1 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx-1; i >= 0; i ++) {
            if (visible_nodes[i+1] - visible_nodes[i] == 1) {
                consecutive_visible_nodes_1.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        last_found_index = alignment_node_idx;
        seg_dist_it = visible_nodes[alignment_node_idx];
        cur_center = guide_nodes.row(alignment_node_idx);

        // basically pure pursuit lol
        while (last_found_index-1 >= alignment_node_idx-consecutive_visible_nodes_1.size() && seg_dist_it-1 >= 0) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i-1 >= 0; i --) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it - 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it -= 1;
            }
        }
    }

    return node_pairs;
}

void bcpd_tracker::bcpd (MatrixXd X_orig,
                         MatrixXd& Y_hat,
                         double& sigma2,
                         double beta,
                         double lambda,
                         double omega,
                         double kappa,
                         double gamma,
                         int max_iter,
                         double tol,
                         bool use_prev_sigma2,
                         std::vector<MatrixXd> correspondence_priors,
                         double zeta)
{
    // ===== initialization =====
    bool align = true;
    if (correspondence_priors.size() == 0) {
        align = false;
    }

    MatrixXd X = X_orig.replicate(1, 1);
    MatrixXd J = MatrixXd::Zero(Y_hat.rows(), X_orig.rows() + correspondence_priors.size());
    if (align) {
        X = MatrixXd::Zero(X_orig.rows() + correspondence_priors.size(), 3);
        for (int i = 0; i < X_orig.rows(); i ++) {
            X.row(i) = X_orig.row(i);
        }
        for (int i = X_orig.rows(); i < X_orig.rows() + correspondence_priors.size(); i ++) {
            int index = correspondence_priors[i - X_orig.rows()](0, 0);
            X(i, 0) = correspondence_priors[i - X_orig.rows()](0, 1);
            X(i, 1) = correspondence_priors[i - X_orig.rows()](0, 2);
            X(i, 2) = correspondence_priors[i - X_orig.rows()](0, 3);
            std::cout << "index = " << index << std::endl;
            J(index, i) = 1;
        }
    }

    int M = Y_hat.rows();
    int N = X.rows();

    MatrixXd X_flat = X.replicate(1, 1).transpose();
    MatrixXd Y_flat = Y_hat.replicate(1, 1).transpose();
    X_flat.resize(N*3, 1);
    Y_flat.resize(M*3, 1);

    std::cout << X.row(1) << std::endl;
    std::cout << X_flat(3, 0) << ", " << X_flat(4, 0) << ", " << X_flat(5, 0) << std::endl;
    std::cout << Y_hat.row(1) << std::endl;
    std::cout << Y_flat(3, 0) << ", " << Y_flat(4, 0) << ", " << Y_flat(5, 0) << std::endl;

    MatrixXd Y = Y_hat.replicate(1, 1);
    MatrixXd v_hat = MatrixXd::Zero(M, 3);

    MatrixXd big_sigma = MatrixXd::Identity(M, M);
    MatrixXd alpha_m_bracket = MatrixXd::Ones(M, N) / static_cast<double>(M);
    double s = 1;
    MatrixXd R = MatrixXd::Identity(3, 3);
    MatrixXd t = MatrixXd::Zero(3, 1);
    
    // initialize G
    MatrixXd diff_yy = MatrixXd::Zero(M, M);
    MatrixXd diff_yy_sqrt = MatrixXd::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y.row(i) - Y.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y.row(i) - Y.row(j)).norm();
        }
    }
    MatrixXd G = (-diff_yy / (2 * beta * beta)).array().exp();

    // Initialize sigma2
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }
    if (!use_prev_sigma2 || sigma2 == 0) {
        sigma2 = gamma * diff_xy.sum() / static_cast<double>(3 * M * N);
    }
    
    // ===== geodesic distance =====
    MatrixXd converted_node_dis = MatrixXd::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    MatrixXd converted_node_dis_sq = MatrixXd::Zero(M, M);
    std::vector<double> converted_node_coord = {0.0};   // this is not squared
    double cur_sum = 0;
    for (int i = 0; i < M-1; i ++) {
        cur_sum += pt2pt_dis(Y.row(i+1), Y.row(i));
        converted_node_coord.push_back(cur_sum);
    }

    for (int i = 0; i < converted_node_coord.size(); i ++) {
        for (int j = 0; j < converted_node_coord.size(); j ++) {
            converted_node_dis_sq(i, j) = pow(converted_node_coord[i] - converted_node_coord[j], 2);
            converted_node_dis(i, j) = abs(converted_node_coord[i] - converted_node_coord[j]);
        }
    }
    G = (-converted_node_dis / (2 * beta * beta)).array().exp();

    // ===== log time and initial values =====
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    MatrixXd prev_Y_hat = Y_hat.replicate(1, 1);
    double prev_sigma2 = sigma2;

    MatrixXd Y_hat_flat = Y_hat.replicate(1, 1).transpose();
    Y_hat_flat.resize(M*3, 1);
    MatrixXd v_hat_flat = v_hat.replicate(1, 1).transpose();
    v_hat_flat.resize(M*3, 1);

    for (int it = 0; it < max_iter; it ++) {
        std::cout << "---- iteration -----" << std::endl;
        std::cout << it << std::endl;

        // ===== update P and related terms =====
        diff_xy = MatrixXd::Zero(M, N);
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y_hat.row(i) - X.row(j)).squaredNorm();
            }
        }
        MatrixXd phi_mn_bracket_1 = pow(2.0*M_PI*sigma2, -3.0/2.0) * (1.0-omega) * (-0.5 * diff_xy / sigma2).array().exp();  // this is M by N
        MatrixXd phi_mn_bracket_2 = (-pow(s, 2) / (2*sigma2) * 3 * big_sigma.diagonal()).array().exp();  // this is M by 1 or 1 by M. more likely 1 by M
        
        phi_mn_bracket_2.resize(M, 1);
        MatrixXd phi_mn_bracket_2_expanded = phi_mn_bracket_2.replicate(1, N);  // expand to M by N
        // MatrixXd P = (phi_mn_bracket_1.cwiseProduct(phi_mn_bracket_2_expanded)).cwiseProduct(alpha_m_bracket);
        MatrixXd P = phi_mn_bracket_1.cwiseProduct(alpha_m_bracket);
        double c = omega / N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // P = P.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });

        // MatrixXd P1 = P.rowwise().sum();
        // MatrixXd Pt1 = P.colwise().sum();
        MatrixXd nu = P.rowwise().sum();
        MatrixXd nu_prime = P.colwise().sum();
        double N_hat = P.sum();

        std::cout << "=== nu ===" << std::endl;
        std::cout << nu << std::endl;
        std::cout << "=== nu_corr ===" << std::endl;
        std::cout << J.rowwise().sum() << std::endl;

        // compute X_hat
        MatrixXd nu_tilde = Eigen::kroneckerProduct(nu, MatrixXd::Constant(3, 1, 1.0));
        MatrixXd P_tilde = Eigen::kroneckerProduct(P, MatrixXd::Identity(3, 3));
        MatrixXd X_hat_flat = (nu_tilde.asDiagonal().inverse()) * (P_tilde * X_flat);

        for (int m = 0; m < 3*M; m ++) {
            if (nu_tilde(m, 0) == 0.0) {
                X_hat_flat(m, 0) = 0.0;
            }
        }

        MatrixXd X_hat_t = X_hat_flat.replicate(1, 1);
        X_hat_t.resize(3, M);
        MatrixXd X_hat = X_hat_t.transpose();

        // std::cout << "=== X_hat_flat ===" << std::endl;
        // std::cout << X_hat_flat << std::endl;
        // std::cout << "=== X_hat_t ===" << std::endl;
        // std::cout << X_hat_t << std::endl;
        // std::cout << "=== X_hat ===" << std::endl;
        // std::cout << X_hat << std::endl;

        // ===== update big_sigma, v_hat, u_hat, and alpha_m_bracket for all m =====
        if (!align) {
            big_sigma = lambda * G.inverse();
            big_sigma += pow(s, 2)/sigma2 * nu.asDiagonal();
            big_sigma = big_sigma.inverse();
            MatrixXd big_sigma_tilde = Eigen::kroneckerProduct(big_sigma, MatrixXd::Identity(3, 3));
            MatrixXd R_tilde = Eigen::kroneckerProduct(R, MatrixXd::Identity(M, M));
            MatrixXd t_tilde = Eigen::kroneckerProduct(MatrixXd::Constant(M, 1, 1.0), t);

            MatrixXd residual = 1/s * R_tilde.transpose() * (X_hat_flat - t_tilde) - Y_flat;
            v_hat_flat = pow(s, 2) / sigma2 * big_sigma_tilde * nu_tilde.asDiagonal() * residual;
            MatrixXd v_hat_t = v_hat_flat.replicate(1, 1);
            v_hat_t.resize(3, M);
            v_hat = v_hat_t.transpose();
        }
        else {
            std::cout << "in else statment" << std::endl;

            MatrixXd nu_corr = J.rowwise().sum();
            MatrixXd nu_corr_tilde = Eigen::kroneckerProduct(nu_corr, MatrixXd::Constant(3, 1, 1.0));
            MatrixXd J_tilde = Eigen::kroneckerProduct(J, MatrixXd::Identity(3, 3));

            std::cout << "after calculating J and nu tilde" << std::endl;
            
            big_sigma = lambda * G.inverse();
            big_sigma += pow(s, 2)/sigma2 * nu.asDiagonal();
            big_sigma += pow(s, 2)/zeta * nu_corr.asDiagonal();
            big_sigma = big_sigma.inverse();

            std::cout << "after calculating big_sigma" << std::endl;

            MatrixXd big_sigma_tilde = Eigen::kroneckerProduct(big_sigma, MatrixXd::Identity(3, 3));
            MatrixXd R_tilde = Eigen::kroneckerProduct(R, MatrixXd::Identity(M, M));
            MatrixXd t_tilde = Eigen::kroneckerProduct(MatrixXd::Constant(M, 1, 1.0), t);

            std::cout << "before calculating residual" << std::endl;

            MatrixXd residual = 1/s * R_tilde.transpose() * (X_hat_flat - t_tilde) - Y_flat;
            MatrixXd dv_residual = nu_corr_tilde.asDiagonal() * (1/s * R_tilde.transpose() * J_tilde * X_flat - 1/s * R_tilde.transpose() * t_tilde - Y_flat);
            v_hat_flat = pow(s, 2) / sigma2 * big_sigma_tilde * nu_tilde.asDiagonal() * residual + pow(s, 2) / zeta * big_sigma_tilde * dv_residual;
            MatrixXd v_hat_t = v_hat_flat.replicate(1, 1);
            v_hat_t.resize(3, M);
            v_hat = v_hat_t.transpose();
        }

        MatrixXd u_hat = Y + v_hat;
        MatrixXd u_hat_flat = Y_flat + v_hat_flat;

        MatrixXd alpha_m_bracket_1 = MatrixXd::Constant(nu.rows(), nu.cols(), kappa) + nu;
        MatrixXd alpha_m_bracket_2 = MatrixXd::Constant(nu.rows(), nu.cols(), kappa*M + N_hat);
        MatrixXd alpha_m_bracket_1_digamma = Eigen::digamma(alpha_m_bracket_1.array());
        MatrixXd alpha_m_bracket_2_digamma = Eigen::digamma(alpha_m_bracket_2.array());
        MatrixXd alpha_m_bracket_no_expansion = (alpha_m_bracket_1_digamma - alpha_m_bracket_2_digamma).array().exp();
        alpha_m_bracket_no_expansion = alpha_m_bracket_no_expansion.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
        alpha_m_bracket_no_expansion.resize(M, 1);
        alpha_m_bracket = alpha_m_bracket_no_expansion.replicate(1, N);

        // ===== update s, R, t, sigma2, y_hat =====
        // nu is M by 1
        MatrixXd nu_expanded = nu.replicate(1, 3);
        // MatrixXd X_bar = (nu_expanded.cwiseProduct(X_hat)).colwise().sum() / N_hat;
        // MatrixXd u_bar = (nu_expanded.cwiseProduct(u_hat)).colwise().sum() / N_hat;

        // X_bar is 1 by 3
        double sigma2_bar = (nu.cwiseProduct(big_sigma.diagonal())).sum() / N_hat;

        // MatrixXd S_xu = MatrixXd::Zero(3, 3);
        // MatrixXd S_uu = MatrixXd::Zero(3, 3);
        // for (int m = 0; m < M; m ++) {
        //     MatrixXd X_diff = X_hat.row(m) - X_bar;
        //     MatrixXd u_diff = u_hat.row(m) - u_bar;
        //     X_diff.resize(3, 1);
        //     u_diff.resize(1, 3);
        //     S_xu += nu(m, 0) * (X_diff * u_diff);
        //     S_uu += nu(m, 0) * (u_diff.transpose() * u_diff);
        // }
        // S_xu /= N_hat;
        // S_uu /= N_hat;
        // S_uu += sigma2_bar * MatrixXd::Identity(3, 3);
        // Eigen::JacobiSVD<Eigen::MatrixXd> svd(S_xu, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // MatrixXd U = svd.matrixU();
        // MatrixXd S = svd.singularValues();
        // MatrixXd V = svd.matrixV();
        // MatrixXd Vt = V.transpose();
        // MatrixXd middle_mat = MatrixXd::Identity(3, 3);
        // middle_mat(2, 2) = (U * Vt).determinant();
        // R = U * middle_mat * Vt;

        // s = (R * S_xu).trace() / S_uu.trace();
        // t = X_bar.transpose() - s*R*u_bar.transpose();

        MatrixXd T_hat = MatrixXd::Identity(4, 4);
        T_hat.block<3, 3>(0, 0) = s*R;
        T_hat.block<3, 1>(0, 3) = t;

        MatrixXd Y_hat_h = u_hat.replicate(1, 1);
        Y_hat_h.conservativeResize(Y_hat.rows(), Y_hat.cols()+1);
        Y_hat_h.col(Y_hat_h.cols()-1) = MatrixXd::Ones(Y_hat_h.rows(), 1);
        Y_hat = (T_hat * Y_hat_h.transpose()).transpose().leftCols(3);
        Y_hat_flat = Y_hat.replicate(1, 1).transpose();
        Y_hat_flat.resize(M*3, 1);

        MatrixXd nu_prime_tilde = Eigen::kroneckerProduct(nu_prime, MatrixXd::Constant(1, 3, 1.0));
        MatrixXd sigma2_mat = 1/(N_hat*3) * (X_flat.transpose()*nu_prime_tilde.asDiagonal()*X_flat - 2*X_flat.transpose()*P_tilde.transpose()*Y_hat_flat + Y_hat_flat.transpose()*nu_tilde.asDiagonal()*Y_hat_flat) + pow(s, 2) * MatrixXd::Constant(1, 1, sigma2_bar);
        sigma2 = abs(sigma2_mat(0, 0));

        std::cout << "=== sigma2 ===" << std::endl;
        std::cout << sigma2_mat << std::endl;

        // ===== check convergence =====
        if (fabs(sigma2 - prev_sigma2) < tol && (Y_hat - prev_Y_hat).cwiseAbs().maxCoeff() < tol) {
            ROS_INFO_STREAM(("Converged after " + std::to_string(it) + " iterations. Time taken: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count()) + " ms."));
            break;
        }

        if (it == max_iter - 1) {
            ROS_ERROR_STREAM(("Optimization did not converge! Time taken: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count()) + " ms."));
        }

        prev_Y_hat = Y_hat.replicate(1, 1);
        prev_sigma2 = sigma2;
    }
}

void bcpd_tracker::tracking_step (MatrixXd X_orig,
                                  Mat bmask_transformed_normalized,
                                  double mask_dist_threshold,
                                  double mat_max) {
    
    // variable initialization
    std::vector<int> occluded_nodes = {};
    std::vector<int> visible_nodes = {};
    std::vector<MatrixXd> valid_nodes_vec = {};
    correspondence_priors_ = {};
    int state = 0;

    // project Y onto the original image to determine occluded nodes
    MatrixXd nodes_h = Y_.replicate(1, 1);
    nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
    nodes_h.col(nodes_h.cols()-1) = MatrixXd::Ones(nodes_h.rows(), 1);
    MatrixXd proj_matrix(3, 4);
    proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                    0.0, 916.265869140625, 354.02392578125, 0.0,
                    0.0, 0.0, 1.0, 0.0;
    MatrixXd image_coords = (proj_matrix * nodes_h.transpose()).transpose();
    for (int i = 0; i < image_coords.rows(); i ++) {
        int x = static_cast<int>(image_coords(i, 0)/image_coords(i, 2));
        int y = static_cast<int>(image_coords(i, 1)/image_coords(i, 2));

        // not currently using the original distance transform because I can't figure it out
        if (static_cast<int>(bmask_transformed_normalized.at<uchar>(y, x)) < mask_dist_threshold / mat_max * 255) {
            valid_nodes_vec.push_back(Y_.row(i));
            visible_nodes.push_back(i);
        }
        else {
            occluded_nodes.push_back(i);
        }
    }

    // copy valid guide nodes vec to guide nodes
    // not using topRows() because it caused weird bugs
    guide_nodes_ = MatrixXd::Zero(valid_nodes_vec.size(), 3);
    if (occluded_nodes.size() != 0) {
        for (int i = 0; i < valid_nodes_vec.size(); i ++) {
            guide_nodes_.row(i) = valid_nodes_vec[i];
        }
    }
    else {
        guide_nodes_ = Y_.replicate(1, 1);
    }

    // determine DLO state: heading visible, tail visible, both visible, or both occluded
    // priors_vec should be the final output; priors_vec[i] = {index, x, y, z}
    double sigma2_pre_proc = sigma2_;
    cpd_lle(X_orig, guide_nodes_, sigma2_pre_proc, 1, 1, 10, 0.1, 50, 0.00001, true, true, true);

    if (occluded_nodes.size() == 0) {
        ROS_INFO("All nodes visible");

        // // get priors vec
        // std::vector<MatrixXd> priors_vec_1 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
        // std::vector<MatrixXd> priors_vec_2 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 1);
        // // std::vector<MatrixXd> priors_vec_1 = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 0);
        // // std::vector<MatrixXd> priors_vec_2 = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);

        // // take average
        // correspondence_priors_ = {};
        // for (int i = 0; i < Y_.rows(); i ++) {
        //     if (i < priors_vec_2[0](0, 0) && i < priors_vec_1.size()) {
        //         correspondence_priors_.push_back(priors_vec_1[i]);
        //     }
        //     else if (i > priors_vec_1[priors_vec_1.size()-1](0, 0) && (i-(Y_.rows()-priors_vec_2.size())) < priors_vec_2.size()) {
        //         correspondence_priors_.push_back(priors_vec_2[i-(Y_.rows()-priors_vec_2.size())]);
        //     }
        //     else {
        //         correspondence_priors_.push_back((priors_vec_1[i] + priors_vec_2[i-(Y_.rows()-priors_vec_2.size())]) / 2.0);
        //     }
        // }

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
    }
    else if (visible_nodes[0] == 0 && visible_nodes[visible_nodes.size()-1] == Y_.rows()-1) {
        ROS_INFO("Mid-section occluded");

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
        std::vector<MatrixXd> priors_vec_2 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 1);
        // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 0);
        // std::vector<MatrixXd> priors_vec_2 = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);

        correspondence_priors_.insert(correspondence_priors_.end(), priors_vec_2.begin(), priors_vec_2.end());
    }
    else if (visible_nodes[0] == 0) {
        ROS_INFO("Tail occluded");

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
        // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 0);
    }
    else if (visible_nodes[visible_nodes.size()-1] == Y_.rows()-1) {
        ROS_INFO("Head occluded");

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 1);
        // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);
    }
    else {
        ROS_INFO("Both ends occluded");

        // determine which node moved the least
        int alignment_node_idx = -1;
        double moved_dist = 999999;
        for (int i = 0; i < visible_nodes.size(); i ++) {
            if (pt2pt_dis(Y_.row(visible_nodes[i]), guide_nodes_.row(i)) < moved_dist) {
                moved_dist = pt2pt_dis(Y_.row(visible_nodes[i]), guide_nodes_.row(i));
                alignment_node_idx = i;
            }
        }

        // std::cout << "alignment node index: " << alignment_node_idx << std::endl;
        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 2, alignment_node_idx);
    }

    std::cout << "finished traversal" << std::endl;
    for (int i = 0; i < correspondence_priors_.size(); i ++) {
        std::cout << correspondence_priors_[i] << std::endl;
    }

    // include_lle == false because we have no space to discuss it in the paper
    // ecpd_lle (X_orig, Y_, sigma2_, beta_, lambda_, lle_weight_, mu_, max_iter_, tol_, include_lle_, use_geodesic_, use_prev_sigma2_, true, correspondence_priors_, alpha_, kernel_, occluded_nodes, k_vis_, bmask_transformed_normalized, mat_max);
    bcpd(X_orig, Y_, sigma2_, beta_, lambda_, omega_, kappa_, gamma_, max_iter_, tol_, use_prev_sigma2_, correspondence_priors_, zeta_);
}