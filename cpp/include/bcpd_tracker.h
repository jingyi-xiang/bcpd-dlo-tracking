
#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SpecialFunctions>
#include <vector>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/rgbd.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float64.h>

#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm> 

#include <unistd.h>
#include <cstdlib>
#include <signal.h>


#ifndef BCPD_TRACKER_H
#define BCPD_TRACKER_H

using Eigen::MatrixXd;
using cv::Mat;

class bcpd_tracker
{
    public:
        // default constructor
        bcpd_tracker();
        bcpd_tracker(int num_of_nodes);
        // fancy constructor
        bcpd_tracker(int num_of_nodes,
                     double beta,
                     double lambda,
                     double omega,
                     double kappa,
                     double gamma,
                     int max_iter,
                     const double tol,
                     bool use_prev_sigma2);

        double get_sigma2();
        MatrixXd get_tracking_result();
        void initialize_nodes (MatrixXd Y_init);
        void set_sigma2 (double sigma2);

        // ===== Parameters =====
        // X \in R^N  -- target point set
        // Y \in R^M  -- source point set 
        // beta       -- controls the influence of motion coherence
        // omega      -- the outlier probability
        // kappa      -- the parameter of the Dirichlet distribution used as a prior distribution of alpha
        // gamma      -- the scale factor of sigma2_0
        void bcpd (MatrixXd X,
                   MatrixXd& Y_hat,
                   double& sigma2,
                   double beta,
                   double lambda,
                   double omega,
                   double kappa,
                   double gamma,
                   int max_iter = 50,
                   double tol = 0.00001,
                   bool use_prev_sigma2 = false);

    private:
        MatrixXd Y_;
        double sigma2_;
        double beta_;
        double lambda_;
        double omega_;
        double kappa_;
        double gamma_;
        int max_iter_;
        double tol_;
        bool use_prev_sigma2_;
};

#endif