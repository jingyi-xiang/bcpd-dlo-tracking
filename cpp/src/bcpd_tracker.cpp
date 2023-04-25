#include "../include/utils.h"
#include "../include/bcpd_tracker.h"

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;
using cv::Mat;

bcpd_tracker::bcpd_tracker () {}

bcpd_tracker::bcpd_tracker(int num_of_nodes) 
{
    // default initialize
    Y_ = MatrixXf::Zero(num_of_nodes, 3);
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
    Y_ = MatrixXf::Zero(num_of_nodes, 3);
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

MatrixXf bcpd_tracker::get_tracking_result () {
    return Y_;
}

void bcpd_tracker::initialize_nodes (MatrixXf Y_init) {
    Y_ = Y_init.replicate(1, 1);
}

void bcpd_tracker::set_sigma2 (double sigma2) {
    sigma2_ = sigma2;
}

void bcpd_tracker::bcpd (MatrixXf X,
                         MatrixXf& Y_hat,
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

    MatrixXf X_flat = X.replicate(1, 1);
    X_flat.resize(N*3, 1);
    MatrixXf Y_flat = Y_hat.replicate(1, 1);
    Y_flat.resize(M*3, 1);

    MatrixXf Y = Y_hat.replicate(1, 1);
    MatrixXf v_hat = MatrixXf::Zero(M, 3);

    MatrixXf big_sigma = MatrixXf::Identity(M, M);
    MatrixXf alpha_m_bracket = MatrixXf::Ones(M, N) / M;
    double s = 1;
    MatrixXf R = MatrixXf::Identity(3, 3);
    MatrixXf t = MatrixXf::Zero(1, 3);
    
    // initialize G
    MatrixXf diff_yy = MatrixXf::Zero(M, M);
    MatrixXf diff_yy_sqrt = MatrixXf::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y.row(i) - Y.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y.row(i) - Y.row(j)).norm();
        }
    }
    MatrixXf G = (-diff_yy / (2 * beta * beta)).array().exp();

    // Initialize sigma2
    MatrixXf diff_xy = MatrixXf::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }
    if (!use_prev_sigma2 || sigma2 == 0) {
        sigma2 = diff_xy.sum() / static_cast<double>(3 * M * N);
    }

    // ===== log time and initial values =====
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    MatrixXf prev_Y_hat = Y_hat.replicate(1, 1);
    double prev_sigma2 = sigma2;

    for (int i = 0; i < max_iter; i ++) {
        MatrixXf Y_hat_flat = Y_hat.replicate(1, 1);
        Y_hat_flat.resize(M*3, 1);
        MatrixXf v_hat_flat = v_hat.replicate(1, 1);
        v_hat_flat.resize(M*3, 1);

        // ===== update P and related terms =====
        diff_xy = MatrixXf::Zero(M, N);
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }
        MatrixXf phi_mn_bracket_1 = (-0.5 * diff_xy / sigma2).array().exp() * pow(2*M_PI*sigma2, -3.0/2.0) * (1-omega);  // this is M by N
        MatrixXf phi_mn_bracket_2 = (-pow(s, 2) / (2*sigma2) * 3 * big_sigma.diagonal()).array().exp();  // this is M by 1 or 1 by M. more likely 1 by M
        phi_mn_bracket_2.resize(M, 1);
        phi_mn_bracket_2 = phi_mn_bracket_2.replicate(1, N);  // expand to M by N
        MatrixXf P = (phi_mn_bracket_1.cwiseProduct(phi_mn_bracket_2)).cwiseProduct(alpha_m_bracket);
        double c = omega / N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        // MatrixXf P1 = P.rowwise().sum();
        // MatrixXf Pt1 = P.colwise().sum();
        MatrixXf nu = P.rowwise().sum();
        MatrixXf nu_prime = P.colwise().sum();
        double N_hat = P.sum();

        // compute X_hat
        MatrixXf nu_tilde = Eigen::kroneckerProduct(nu, MatrixXf::Identity(3, 3));
        MatrixXf P_tilde = Eigen::kroneckerProduct(P, MatrixXf::Identity(3, 3));
        MatrixXf X_hat_flat = nu_tilde.asDiagonal().inverse() * P_tilde * X_flat;
        MatrixXf X_hat = nu.asDiagonal().inverse() * P * X;

        // ===== update big_sigma, v_hat, u_hat, and alpha_m_bracket for all m =====
        big_sigma = lambda * G.inverse();
        big_sigma += pow(s, 2)/sigma2 * nu.asDiagonal();
        big_sigma = big_sigma.inverse();
        MatrixXf T = MatrixXf::Identity(4, 4);
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;
        MatrixXf T_inv = T.inverse();

        MatrixXf X_hat_h = X_hat.replicate(1, 1);
        X_hat_h.conservativeResize(X_hat_h.rows(), X_hat_h.cols()+1);
        X_hat_h.col(X_hat_h.cols()-1) = MatrixXf::Ones(X_hat_h.rows(), 1);
        MatrixXf Y_h = Y.replicate(1, 1);
        Y_h.conservativeResize(Y.rows(), Y_h.cols()+1);
        Y_h.col(Y_h.cols()-1) = MatrixXf::Ones(Y_h.rows(), 1);
        MatrixXf residual= ((T_inv * X_hat_h.transpose()).transpose() - Y_h).leftCols(3);
        MatrixXf v_hat = pow(s, 2)/sigma2 * big_sigma * nu.asDiagonal() * residual;
        v_hat_flat = v_hat.replicate(1, 1);
        v_hat_flat.resize(v_hat.rows()*v_hat.cols(), 1);

        MatrixXf u_hat = Y + v_hat;
        MatrixXf u_hat_flat = Y_flat + v_hat_flat;

        MatrixXf alpha_m_bracket_1 = MatrixXf::Constant(nu.rows(), nu.cols(), kappa) + nu;
        MatrixXf alpha_m_bracket_2 = MatrixXf::Constant(nu.rows(), nu.cols(), kappa*M + N_hat);
        alpha_m_bracket_1 = Eigen::digamma(alpha_m_bracket_1.array());
        alpha_m_bracket_2 = Eigen::digamma(alpha_m_bracket_2.array());
        alpha_m_bracket = (alpha_m_bracket_1 - alpha_m_bracket_2).array().exp();
        alpha_m_bracket.resize(M, 1);
        alpha_m_bracket = alpha_m_bracket.replicate(1, N);

        // ===== update s, R, t, sigma2, y_hat =====
        MatrixXf X_bar = (nu.replicate(1, 3) * X_hat).colwise().sum() / N_hat;
        MatrixXf u_bar = (nu.replicate(1, 3) * u_hat).colwise().sum() / N_hat;
        double sigma2_bar = (nu*sigma2).sum() / N_hat;

        MatrixXf S_xu = MatrixXf::Zero(3, 3);
        MatrixXf S_uu = MatrixXf::Zero(3, 3);
        for (int m = 0; m < M; m ++) {
            MatrixXf X_diff = X_hat.row(m) - X_bar;
            MatrixXf u_diff = u_hat.row(m) - u_bar;
            X_diff.resize(3, 1);
            u_diff.resize(1, 3);
            S_xu += nu.row(m) * (X_diff * u_diff);
            S_uu += nu.row(m) * (u_diff.transpose() * u_diff);
        }
        S_xu /= N_hat;
        S_uu /= N_hat;
        S_uu += sigma2_bar * MatrixXf::Identity(3, 3);
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(S_xu, Eigen::ComputeFullU | Eigen::ComputeFullV);
        MatrixXf U = svd.matrixU();
        MatrixXf S = svd.singularValues();
        MatrixXf V = svd.matrixV();
        MatrixXf Vt = V.transpose();
        MatrixXf middle_mat = MatrixXf::Identity(3, 3);
        middle_mat(2, 2) = (U * V).determinant();
        R = U * middle_mat * Vt;

        s = (R * S_xu).trace() / S_uu.trace();
        t = X_bar - s*R*u_bar;

        MatrixXf T_hat = MatrixXf::Identity(4, 4);
        T_hat.block<3, 3>(0, 0) = s*R;
        T_hat.block<3, 1>(0, 3) = t;
        MatrixXf Y_hat_h = Y_hat.replicate(1, 1);
        Y_hat_h.conservativeResize(Y_hat.rows(), Y_hat.cols()+1);
        Y_hat_h.col(Y_hat_h.cols()-1) = MatrixXf::Ones(Y_hat_h.rows(), 1);
        Y_hat = (T_hat * Y_hat_h.transpose()).leftCols(3).transpose();
    
        MatrixXf nu_prime_tilde = Eigen::kroneckerProduct(nu_prime, MatrixXf::Constant(1, 3, 1.0));
        MatrixXf sigma2_mat = 1/(N_hat*3) * (X_flat.transpose()*nu_prime_tilde.asDiagonal()*X_flat - 2*X_flat.transpose()*P_tilde*Y_hat_flat + Y_hat_flat.transpose()*nu_tilde.asDiagonal()) + pow(s, 2) * MatrixXf::Constant(1, 1, sigma2_bar);
        sigma2 = sigma2_mat(0, 0);

        // ===== check convergence =====
        if (fabs(sigma2 - prev_sigma2) < tol && (Y_hat - prev_Y_hat).cwiseAbs().maxCoeff() < tol) {
            ROS_INFO_STREAM(("Converged after " + std::to_string(i) + " iterations. Time taken: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count()) + " ms."));
            break;
        }

        if (i == max_iter - 1) {
            ROS_ERROR_STREAM(("Optimization did not converge! Time taken: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count()) + " ms."));
        }

        prev_Y_hat = Y_hat.replicate(1, 1);
        prev_sigma2 = sigma2;
    }
}