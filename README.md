# [Baxter autonomous gripper control](https://emarolab.github.io/baxter_autonomous_gripper_control/)
This package collects ROS nodes for the Baxter autonomous gripper control and python scripts useful to design a predictive model for the Baxter's gripper behavior.


## Installation

Clone the repository in the source folder of your catkin workspace.

    git clone https://github.com/EmaroLab/baxter_autonomous_gripper_control.git


#### Prerequisites and Dependecies

In order to succesfully run the code, you should have installed [ROS](http://wiki.ros.org/kinetic/Installation/Ubuntu) and [Tensorflow](https://www.tensorflow.org/install) on your Ubuntu distribution.


## The project

This project has the main goal in designing a ROS node for the Baxter Research Robot, to let the robot predict gripper position during a 'pick and place' task.
For the purpose, inertial data, from a smartwatch and Baxter's joints, has been collected and combined to fit an LSTM recurrent neural network.

#### In this repository:
* **_src_** folder
  * record_data.py &nbsp;:arrow_right:&nbsp;  _ROS node to collect data coming from the smartwatch and Baxter_
  * auto_grasping.py &nbsp;:arrow_right:&nbsp; _ROS node to predict in real time the gripper position_  
* **_scripts_** folder
  * RawData_Interpolate.py  &nbsp;:arrow_right:&nbsp; _Python scripts to interpolate Baxter data with smartwatch timesteps_
  * PCA_SplinedData.py &nbsp;:arrow_right:&nbsp; _Python scripts to do a principal component analysis on the dataset_
  * RNN_baxter_grasping.py &nbsp;:arrow_right:&nbsp; _Python scripts to create time series and to fit a Keras LSTM sequential model_
* **_model_** folder
  * model.h5 &nbsp;:arrow_right:&nbsp;  _The chosen model from the RNN_baxter_graspimg.py and used by auto_grasping.py node_ 
  * mean_std_zscore &nbsp;:arrow_right:&nbsp; _File .txt where useful data for the normalization are written_
  

#### Step1: Autonomous grasping
**_Currently_**, this study has reached the goal of the autonomous grasping with the following conclusions:
  * The Baxter grasping action does not depend on the position of the object on a flat surface
  * The Baxter grasping action can be predicted using objects of different shape associated with different ways of grabbing      (e.g. a glass grabbed from the side or a ball grabbed from above)
  * It is not possible, with used technologies, to discriminate between the grabbing action and the release action without       manipulating raw data such that: every time sequence of the dataset take into account the initial state of the gripper       for that sequence.
> A demonstration is available at [LINK VIDEO] ___


## Documentation

[emarolab.github.io/baxter_autonomous_gripper_control/](https://emarolab.github.io/baxter_autonomous_gripper_control/)

