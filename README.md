# bcpd-dlo-tracking

This repository contains the work-in-progress code for my Spring 2023 CS 498 Machine Perception final project, wrapped into a ROS (Robot Operating System) package. The goal of the project is to implement papers [*A Bayesian Formulation of Coherent Point Drift*](https://ieeexplore.ieee.org/abstract/document/8985307) and (partially) [*Geodesic-Based Bayesian Coherent Point Drift*](https://ieeexplore.ieee.org/abstract/document/9918058) for real-time deformable object shape tracking. Most of the development is first done in Python, then ported to C++ for running in real time.

## Minimum Requirements
* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [Point Cloud Library](https://pointclouds.org/)
* [OpenCV](https://opencv.org/releases/)
* [Open3D](http://www.open3d.org/)

## Other Requirements
* [librealsense](https://github.com/IntelRealSense/librealsense) and [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) (for testing with RGB-D camera stream)
* [Numpy](https://numpy.org/), [Scipy](https://scipy.org/), and [ros_numpy](https://github.com/eric-wieser/ros_numpy) (for running the Python ROS nodes)
* [vedo](https://vedo.embl.es/) (used in `bcpd.py` for visualizing results)
* [pickle](https://docs.python.org/3/library/pickle.html) (used in `bcpd.py` for loading recorded data)

## Usage

First, clone the repository into a ROS workspace and build the package:
```bash
git clone https://github.com/jingyi-xiang/bcpd-dlo-tracking.git
catkin build
```
All parameters for the BCPD algorithm are configurable in `launch/bcpd_tracker.launch` (only for C++). Rebuilding the package is not required for the parameter modifications to take effect. However, `catkin build` is required after modifying any C++ files.

### Testing with RGB-D Camera Stream

This package has been tested with a Intel RealSense D435 camera. The exact camera configurations used are provided in `/config/preset_decimation_4.0_depth_step_100.json` and can be loaded into the camera using the launch files from `realsense-ros`. Run the following commands to start the realsense camera and the BCPD tracking node:
```bash
roslaunch bcpd-dlo-tracking realsense_node.launch
roslaunch bcpd-dlo-tracking bcpd_tracker.launch
```

### Testing with Provided Data

Recorded frames of point cloud are provided under `/testing_scripts/data/frames` and can be loaded using pickle. Inside the folder, 12 frames of point cloud (`000_pc.json` to `011_pc.json`) and the first 7 frames of corresponding nodes (under folder `/nodes`, `000_nodes.json` to `006_nodes.json`) are provided. To test BCPD with these recorded frames, run the following commands:
```bash
cd testing_scripts
python3 bcpd.py
```

Additionally, we provide ROS bag files for testing with recorded RGB-D stream. The bag files can be found [here](https://drive.google.com/drive/folders/1YjX-xfbNfm_G9FYbdw1voYxmd9VA-Aho?usp=sharing). To test BCPD with these bag files, first download them and place them inside your ROS workspace. Then, run the following commands:
```
roslaunch bcpd-dlo-tracking replay_bag.launch
roslaunch bcpd-dlo-tracking bcpd_tracker.launch
rosbag play <path_to_bag_file>
```
