#include "../include/bcpd_tracker.h"
#include "../include/utils.h"

using cv::Mat;

ros::Publisher pc_pub;
ros::Publisher results_pub;
ros::Publisher guide_nodes_pub;
ros::Publisher result_pc_pub;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;

MatrixXf Y;
double sigma2;
bool initialized = false;
Mat occlusion_mask;
bool updated_opencv_mask = false;

double total_len = 0;
bool visualize_dist = false;

bool use_eval_rope;
int bag_file;
int num_of_nodes;
double beta;
double lambda;
double omega;
double kappa;
double gam;
int max_iter;
double tol;
bool use_prev_sigma2;
double downsample_leaf_size;

int main(int argc, char **argv) {
    ros::init(argc, argv, "bcpd_tracking_node");
    ros::NodeHandle nh;

    ros::spin();
}