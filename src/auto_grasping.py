#!/usr/bin/env python

## @package auto_grasping
# This is a ROS node to predict and command the Baxter gripper (left) position.
# It uses a Tensorflow.Keras LSTM model to predict the gripper position at the end of a time sequence.

import threading
import numpy as np
import math

import rospkg
import rospy
from sensor_msgs.msg import JointState
from mqtt_ros_bridge.msg import ImuPackage
from mqtt_ros_bridge.msg import Vector3Time
from std_msgs.msg import String
from baxter_interface import Limb
from baxter_interface import Gripper

from tensorflow.keras.models import load_model

pkg_dir = rospkg.RosPack().get_path('baxter_autonomous_gripper_control')

av_remains_before = []
la_remains_before = []
watch_timesteps = [0,0] #has 2 values: a list and a flag: 1 to write 0 if writed
is_first_pack = True
firt_pack_sync = threading.Condition()

## Function to manage ramains data in packets
# The smartwacth pubilshes data in packets with two lists (angular velocity, linear acceleration) of different length.
# This function manages this difference to avoid data loss
#
# @param av The angular velocity list
# @param la The linear_acceleration list
def manage_remains(av,la):
    global av_remains_before
    global la_remains_before

    avlen=len(av)
    lalen=len(la)

    av_remains = []
    la_remains = []
    if (avlen < lalen) :
        la_remains = la[avlen:]
        av_remains_before = av_remains
        la_remains_before = la_remains
        return avlen
    elif (lalen < avlen) :
        av_remains = av[lalen:]
        av_remains_before = av_remains
        la_remains_before = la_remains
        return lalen
    else :
        return avlen

## Callback function for the "/imu_giver" topic
def watchCallback(data):
    global av_remains_before
    global la_remains_before
    global watch_timesteps
    global is_first_pack
    global firt_pack_sync

    if (is_first_pack):
        with firt_pack_sync:
            firt_pack_sync.notify_all()
        is_first_pack = False
        return

    #saving list of Vector3Time adding what remains from the previous packet
    av = av_remains_before + data.angular_velocity
    la = la_remains_before + data.linear_acceleration

    packlen = manage_remains(av,la)

    watch_timesteps[0] = []
    for i in range (0, packlen):
        step = [av[i].vector.x, av[i].vector.y, av[i].vector.z, \
                la[i].vector.x, la[i].vector.y, la[i].vector.z]
        watch_timesteps[0].append(step)

    watch_timesteps[1]=1


## ROS node function
# It subscribes to "imu_giver" topic, it manages time sequence and it uses the
# Keras LSTM model to predict the gripper position.
def listener():
    global watch_timesteps
    global firt_pack_sync

    seq_len = 25

    rospy.init_node('auto_grasping', anonymous=True, disable_signals=True)

    model = load_model(pkg_dir + '/model/my_model25-94.h5')

    zscore_data = np.loadtxt(pkg_dir + '/model/mean_std_zscore', delimiter=',', ndmin=2)

    left_arm = Limb('left')
    left_gripper = Gripper('left')
    left_gripper.calibrate()

    rate = rospy.Rate(50) # rate
    rospy.Subscriber('/imu_giver', ImuPackage, watchCallback)
    with firt_pack_sync:
        firt_pack_sync.wait()

    opened_timeout=0
    pre_res = 0
    watch_buffer = []
    bax_timesteps = []
    # bax_timesteps and watch_buffer are two buffers to manage sequences
    while not rospy.is_shutdown():
        rate.sleep()

        l_ang = list(left_arm.joint_angles().values())
        l_vel = list(left_arm.joint_velocities().values())
        l_eff = list(left_arm.joint_efforts().values())

        bax_step = l_ang + l_vel + l_eff
        bax_timesteps.append(bax_step)

        if(watch_timesteps[1]):
            watch_buffer.extend(watch_timesteps[0])
            watch_timesteps[1]=0

        if(len(bax_timesteps)>=seq_len and len(watch_buffer)>=seq_len) :
            watch_buffer = watch_buffer[len(watch_buffer)-(seq_len):]
            bax_timesteps = bax_timesteps[len(bax_timesteps)-(seq_len):]
            sequence = []
            for i in range(0,math.floor(seq_len*0.3)) :
                step = watch_buffer.pop(0) + bax_timesteps.pop(0)
                sequence.append(step)
            for i in range(0,math.ceil(seq_len*0.7)) :
                step = watch_buffer[i] + bax_timesteps[i]
                sequence.append(step)

            sequence = np.array(sequence)

            sequence = sequence - zscore_data[0,:]
            sequence = sequence/zscore_data[1,:]

            seq = np.ndarray((1,seq_len,sequence.shape[1]))
            seq[0] = sequence
            res = model.predict(seq)
            res = res[0][0]
            rospy.loginfo(left_gripper.position())

            if (left_gripper.position()>94.0):
                opened_timeout=opened_timeout+1

            if (res > 0.7 and pre_res > 0.7 and opened_timeout > 25) :
                left_gripper.command_position(0.0)
                opened_timeout=0
            pre_res = res

if __name__ == '__main__':
    listener()
