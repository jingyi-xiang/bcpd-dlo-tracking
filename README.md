# bcpd-dlo-tracking

This repository contains the work-in-progress code for my CS 498 Machine Perception final project (Spring 2023). The goal of the project is to implement paper [*A Bayesian Formulation of Coherent Point Drift*](https://ieeexplore.ieee.org/abstract/document/8985307) for real-time deformable object shape tracking and evaluate its performance against existing deformable object shape tracking approaches. Both Python and C++ are used for implementing the algorithm.

## Usage

### To test BCPD with recorded frame data (discrete):
```bash
cd testing_scripts && python3 bcpd.py
```

### To test BCPD with RGBD camera stream:
```bash
# I used a Intel Realsense D435 rgbd camera
roslaunch bcpd-dlo-tracking realsense_node.launch  # terminal 1
rosrun bcpd-dlo-tracking bcpd_tracking_node.py  # terminal 2
```
