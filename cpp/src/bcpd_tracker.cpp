#include "../include/utils.h"
#include "../include/bcpd_tracker.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXf;
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

void bcpd_tracker::initialize_nodes (MatrixXd Y_init) {
    Y_ = Y_init.replicate(1, 1);
}

void bcpd_tracker::set_sigma2 (double sigma2) {
    sigma2_ = sigma2;
}

void bcpd_tracker::bcpd (MatrixXd X,
                         MatrixXd& Y_hat,
                         double& sigma2,
                         double beta,
                         double lambda,
                         double omega,
                         double kappa,
                         double gamma,
                         int max_iter,
                         double tol,
                         bool use_prev_sigma2)
{
    // ===== initialization =====
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

    std::cout << "=== len(Y) ===" << std::endl;
    std::cout << Y.rows() << std::endl;

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
    
    // // ===== geodesic distance =====
    // MatrixXd converted_node_dis = MatrixXd::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    // MatrixXd converted_node_dis_sq = MatrixXd::Zero(M, M);
    // std::vector<double> converted_node_coord = {0.0};   // this is not squared
    // double cur_sum = 0;
    // for (int i = 0; i < M-1; i ++) {
    //     cur_sum += pt2pt_dis(Y.row(i+1), Y.row(i));
    //     converted_node_coord.push_back(cur_sum);
    // }

    // for (int i = 0; i < converted_node_coord.size(); i ++) {
    //     for (int j = 0; j < converted_node_coord.size(); j ++) {
    //         converted_node_dis_sq(i, j) = pow(converted_node_coord[i] - converted_node_coord[j], 2);
    //         converted_node_dis(i, j) = abs(converted_node_coord[i] - converted_node_coord[j]);
    //     }
    // }
    // MatrixXd G = (-converted_node_dis / (2 * beta * beta)).array().exp();

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
        std::cout << "=== diff_xy ===" << std::endl;
        std::cout << (-0.5 * diff_xy.row(0) / sigma2).array().exp().sum() << std::endl;
        MatrixXd phi_mn_bracket_1 = pow(2.0*M_PI*sigma2, -3.0/2.0) * (1.0-omega) * (-0.5 * diff_xy / sigma2).array().exp();  // this is M by N
        MatrixXd phi_mn_bracket_2 = (-pow(s, 2) / (2*sigma2) * 3 * big_sigma.diagonal()).array().exp();  // this is M by 1 or 1 by M. more likely 1 by M
        
        // std::cout << "=== phi_mn_bracket_1 ===" << std::endl;
        // std::cout << phi_mn_bracket_1 << std::endl;
        std::cout << "=== phi_mn_bracket_2 ===" << std::endl;
        std::cout << phi_mn_bracket_2 << std::endl;
        // std::cout << sigma2 << std::endl;
        // std::cout << exp(-pow(s, 2) / (2*sigma2) * 3) << std::endl;
        
        phi_mn_bracket_2.resize(M, 1);
        MatrixXd phi_mn_bracket_2_expanded = phi_mn_bracket_2.replicate(1, N);  // expand to M by N
        MatrixXd P = (phi_mn_bracket_1.cwiseProduct(phi_mn_bracket_2_expanded)).cwiseProduct(alpha_m_bracket);
        double c = omega / N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // P = P.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });

        std::cout << "=== phi_mn_bracket_2_expanded.row(0) ===" << std::endl;
        std::cout << phi_mn_bracket_2_expanded.row(0) << std::endl;
        std::cout << "=== P.rowwise().sum() ===" << std::endl;
        std::cout << P.rowwise().sum() << std::endl;

        // MatrixXd P1 = P.rowwise().sum();
        // MatrixXd Pt1 = P.colwise().sum();
        MatrixXd nu = P.rowwise().sum();
        MatrixXd nu_prime = P.colwise().sum();
        double N_hat = P.sum();
        std::cout << "=== N_hat ===" << std::endl;
        std::cout << N_hat << std::endl; 

        // compute X_hat
        MatrixXd nu_tilde = Eigen::kroneckerProduct(nu, MatrixXd::Constant(1, 3, 1.0));
        MatrixXd P_tilde = Eigen::kroneckerProduct(P, MatrixXd::Identity(3, 3));
        // MatrixXd X_hat_flat = (nu_tilde.asDiagonal().inverse()) * P_tilde * X_flat;
        MatrixXd X_hat = nu.asDiagonal().inverse() * P * X;

        for (int m = 0; m < M; m ++) {
            if (nu(m, 0) == 0.0) {
                X_hat(m, 0) = 0.0;
                X_hat(m, 1) = 0.0;
                X_hat(m, 2) = 0.0;
            }
        }

        // std::cout << "=== X_hat_flat ===" << std::endl;
        // std::cout << X_hat_flat << std::endl;
        std::cout << "=== X_hat ===" << std::endl;
        std::cout << X_hat << std::endl;

        // ===== update big_sigma, v_hat, u_hat, and alpha_m_bracket for all m =====
        big_sigma = lambda * G.inverse();
        big_sigma += pow(s, 2)/sigma2 * nu.asDiagonal();
        big_sigma = big_sigma.inverse();
        MatrixXd T = MatrixXd::Identity(4, 4);
        T.block<3, 3>(0, 0) = s*R;
        T.block<3, 1>(0, 3) = t;
        MatrixXd T_inv = T.inverse();

        MatrixXd X_hat_h = X_hat.replicate(1, 1);
        X_hat_h.conservativeResize(X_hat_h.rows(), X_hat_h.cols()+1);
        X_hat_h.col(X_hat_h.cols()-1) = MatrixXd::Ones(X_hat_h.rows(), 1);
        MatrixXd Y_h = Y.replicate(1, 1);
        Y_h.conservativeResize(Y.rows(), Y_h.cols()+1);
        Y_h.col(Y_h.cols()-1) = MatrixXd::Ones(Y_h.rows(), 1);

        MatrixXd residual= ((T_inv * X_hat_h.transpose()).transpose() - Y_h).leftCols(3);
        v_hat = pow(s, 2)/sigma2 * big_sigma * nu.asDiagonal() * residual;
        v_hat_flat = v_hat.replicate(1, 1).transpose();
        v_hat_flat.resize(M*3, 1);

        MatrixXd u_hat = Y + v_hat;
        MatrixXd u_hat_flat = Y_flat + v_hat_flat;

        // MatrixXd alpha_m_bracket_1 = MatrixXd::Constant(nu.rows(), nu.cols(), kappa) + nu;
        // MatrixXd alpha_m_bracket_2 = MatrixXd::Constant(nu.rows(), nu.cols(), kappa*M + N_hat);
        // alpha_m_bracket_1 = Eigen::digamma(alpha_m_bracket_1.array());
        // alpha_m_bracket_2 = Eigen::digamma(alpha_m_bracket_2.array());
        // alpha_m_bracket = (alpha_m_bracket_1 - alpha_m_bracket_2).array().exp();
        // alpha_m_bracket.resize(M, 1);
        // alpha_m_bracket = alpha_m_bracket.replicate(1, N);

        // ===== update s, R, t, sigma2, y_hat =====
        // nu is M by 1
        MatrixXd nu_expanded = nu.replicate(1, 3);
        std::cout << "=== nu_expanded ===" << std::endl;
        std::cout << nu_expanded << std::endl;
        MatrixXd X_bar = (nu_expanded.cwiseProduct(X_hat)).colwise().sum() / N_hat;
        MatrixXd u_bar = (nu_expanded.cwiseProduct(u_hat)).colwise().sum() / N_hat;
        // X_bar is 1 by 3
        double sigma2_bar = (nu.cwiseProduct(big_sigma.diagonal())).sum() / N_hat;
        std::cout << "=== sigma2_bar ===" << std::endl;
        std::cout << sigma2_bar << std::endl;

        MatrixXd S_xu = MatrixXd::Zero(3, 3);
        MatrixXd S_uu = MatrixXd::Zero(3, 3);
        for (int m = 0; m < M; m ++) {
            MatrixXd X_diff = X_hat.row(m) - X_bar;
            MatrixXd u_diff = u_hat.row(m) - u_bar;
            X_diff.resize(3, 1);
            u_diff.resize(1, 3);
            S_xu += nu(m, 0) * (X_diff * u_diff);
            S_uu += nu(m, 0) * (u_diff.transpose() * u_diff);
        }
        S_xu /= N_hat;
        S_uu /= N_hat;
        S_uu += sigma2_bar * MatrixXd::Identity(3, 3);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S_xu, Eigen::ComputeFullU | Eigen::ComputeFullV);
        MatrixXd U = svd.matrixU();
        MatrixXd S = svd.singularValues();
        MatrixXd V = svd.matrixV();
        MatrixXd Vt = V.transpose();
        MatrixXd middle_mat = MatrixXd::Identity(3, 3);
        middle_mat(2, 2) = (U * Vt).determinant();
        R = U * middle_mat * Vt;

        s = (R * S_xu).trace() / S_uu.trace();
        t = X_bar.transpose() - s*R*u_bar.transpose();

        MatrixXd T_hat = MatrixXd::Identity(4, 4);
        T_hat.block<3, 3>(0, 0) = s*R;
        T_hat.block<3, 1>(0, 3) = t;
        std::cout << "=== T_hat ===" << std::endl;
        std::cout << T_hat << std::endl;

        std::cout << "=== s ===" << std::endl;
        std::cout << s << std::endl;
        // std::cout << "=== R ===" << std::endl;
        // std::cout << R << std::endl;
        // std::cout << "=== RTR ===" << std::endl;
        // std::cout << R.transpose() * R << std::endl;

        MatrixXd Y_hat_h = u_hat.replicate(1, 1);
        Y_hat_h.conservativeResize(Y_hat.rows(), Y_hat.cols()+1);
        Y_hat_h.col(Y_hat_h.cols()-1) = MatrixXd::Ones(Y_hat_h.rows(), 1);
        Y_hat = (T_hat * Y_hat_h.transpose()).transpose().leftCols(3);
        std::cout << "=== Y_hat_h ===" << std::endl;
        std::cout << (T_hat * Y_hat_h.transpose()).transpose() << std::endl;
        std::cout << "=== Y_hat ===" << std::endl;
        std::cout << Y_hat << std::endl;
        Y_hat_flat = Y_hat.replicate(1, 1).transpose();
        Y_hat_flat.resize(M*3, 1);

        std::cout << "=== check Y_hat_flat ===" << std::endl;
        std::cout << Y_hat.row(1) << std::endl;
        std::cout << Y_hat_flat(3, 0) << ", " << Y_hat_flat(4, 0) << ", " << Y_hat_flat(5, 0) << std::endl;

        MatrixXd nu_prime_tilde = Eigen::kroneckerProduct(nu_prime, MatrixXd::Constant(1, 3, 1.0));
        std::cout << "=== nu_prime_tilde ===" << std::endl;
        std::cout << nu_prime_tilde << std::endl;
        std::cout << "=== X_flat.transpose()*nu_prime_tilde.asDiagonal()*X_flat ===" << std::endl;
        std::cout << X_flat.transpose()*nu_prime_tilde.asDiagonal()*X_flat << std::endl;
        std::cout << "=== - 2*X_flat.transpose()*P_tilde.transpose()*Y_hat_flat ===" << std::endl;
        std::cout << - 2*X_flat.transpose()*P_tilde.transpose()*Y_hat_flat << std::endl;
        std::cout << "=== Y_hat_flat.transpose()*nu_tilde.asDiagonal()*Y_hat_flat ===" << std::endl;
        std::cout << Y_hat_flat.transpose()*nu_tilde.asDiagonal()*Y_hat_flat << std::endl;
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