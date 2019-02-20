#!/usr/bin/env python

## @package record_data
# This is a ROS node to collect data simultaneusly from baxter limb and gripprer (left) and a smartwatch.
# It subscribes to the "/imu_giver" topic for the smartwatch data and it uses Baxter interface for the Baxter data.
# It can record sequences using stop (s) and new (n) commands.
#
# It write data for each sequence as follows:
#    - A csv file for Baxter data
#    - A csv file for smartwatch data
# Every row in both files has a timestamp IN NANOSECONDS as last element,
# useful to join data coming from the two different devices.

import os
import sys
import termios
import tty
import csv
from threading import Thread

import rospy
import rospkg
from sensor_msgs.msg import JointState
from mqtt_ros_bridge.msg import ImuPackage
from mqtt_ros_bridge.msg import Vector3Time
from std_msgs.msg import String

from baxter_interface import Limb
from baxter_interface import Gripper

sequence_counter = 0
pkg_dir = rospkg.RosPack().get_path('baxter_autonomous_gripper_control')

watch_file_name = pkg_dir + "/data/raw_sequences/smartwatch_raw_sequence"
WATCH_COLUMNS = 'pack,vx,vy,vz,ax,ay,az,time'

bax_file_name = pkg_dir + "/data/raw_sequences/baxter_raw_sequence"
BAX_COLUMNS = 'angle_left_e0,angle_left_e1,angle_left_s0,angle_left_s1,angle_left_w0,angle_left_w1,angle_left_w2,'+\
        'velocity_left_e0,velocity_left_e1,velocity_left_s0,velocity_left_s1,velocity_left_w0,velocity_left_w1,velocity_left_w2,'+\
        'effort_left_e0,effort_left_e1,effort_left_s0,effort_left_s1,effort_left_w0,effort_left_w1,effort_left_w2,'+\
        'position_l_gripper,time'

NANOS_TO_MILLIS = 1/1000000
packCounter = 0
av_remains_before = []
la_remains_before = []
watch_rows = [0,0] #has 2 values: a list and a flag: 1 to write 0 if not
is_first_pack = True
last_local_time = 0
pack_time_list = []
command = ''

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
    global NANOS_TO_MILLIS
    global av_remains_before
    global la_remains_before
    global watch_rows
    global packCounter
    global is_first_pack
    global last_local_time
    global pack_time_list

    rospy.loginfo("CALLBACK")

    '''
    To maintain the time shift and the differences list, two variables are saved each time a pack arrives:
        - last_local_time : ros timestamp of the last pack
        - last_pack_time : smartwatch timestamp of the last pack

    First pack will be thrown away
    '''
    if (is_first_pack):
        last_local_time = round(rospy.get_rostime().to_nsec()*NANOS_TO_MILLIS)
        last_pack_time = max(data.angular_velocity[-1].time.data, data.linear_acceleration[-1].time.data)
        pack_time_list.append(last_pack_time)
        is_first_pack = False
        return

    packCounter+=1

    #saving list of Vector3Time adding what remains from the previous packet
    av = av_remains_before + data.angular_velocity
    la = la_remains_before + data.linear_acceleration

    packlen = manage_remains(av,la)

    #time shift calculation with a 'diff' list called 'shift'
    for i in range (0, packlen):
        pack_time_list.append(max(av[i].time.data, la[i].time.data))
    shift = [t - s for s, t in zip(pack_time_list, pack_time_list[1:])]

    watch_rows[0] = []
    for i in range (0, packlen):
        t_stamp = last_local_time + sum(shift[0:i+1])
        row=[packCounter, av[i].vector.x, av[i].vector.y, av[i].vector.z, \
            la[i].vector.x, la[i].vector.y, la[i].vector.z, t_stamp]
        watch_rows[0].append(row)
    watch_rows[1]=1

    last_local_time = t_stamp
    last_pack_time = max(av[packlen-1].time.data, la[packlen-1].time.data)
    pack_time_list = []
    pack_time_list.append(last_pack_time)



## Function to get the user input
#
# @return ch The char typed
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        #tty.setraw(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        rospy.loginfo(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

## Thread function to read the user input
def getKey():
    global command
    while not rospy.is_shutdown():
        command = getch()

## Initialization function to start recording a new sequence
def resetNode() :
    global packCounter
    global av_remains_before
    global la_remains_before
    global is_first_pack
    global last_local_time
    global pack_time_list
    global sequence_counter
    global watch_rows

    packCounter = 0
    av_remains_before = []
    la_remains_before = []
    watch_rows = [0,0]
    is_first_pack = True
    last_local_time = 0
    pack_time_list = []

    s_counter = str(sequence_counter)

    with open(watch_file_name + s_counter, 'w') as writeFile:
         writer = csv.writer(writeFile, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
         writer.writerow([WATCH_COLUMNS])
    writeFile.close()

    with open(bax_file_name + s_counter, 'w') as writeFile:
        writer = csv.writer(writeFile, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
        writer.writerow([BAX_COLUMNS])
    writeFile.close()

## ROS node function
# It subscribes to "imu_giver" topic, it collects data from the Baxter Interface, and it writes csv files
def listener():
    rospy.init_node('record_data', anonymous=True, disable_signals=True)

    global BAX_COLUMNS
    global WATCH_COLUMNS
    global NANOS_TO_MILLIS
    global bax_file_name
    global bax_row
    global watch_rows
    global command
    global sequence_counter

    resetNode()

    rospy.loginfo("Commands :\ns to stop\nr to remove the last file\nn to start new sequence\nc TWICE to shutdown the node\n")

    rate = rospy.Rate(120) # rate
    watch_sub = rospy.Subscriber('/imu_giver', ImuPackage, watchCallback)

    rospy.loginfo("START RECORDING SEQUENCE " + str(sequence_counter))

    getkey_thread = Thread(target = getKey)
    getkey_thread.start()

    left_arm = Limb('left')
    left_gripper = Gripper('left')

    while not rospy.is_shutdown():
        rate.sleep()

        t = round(rospy.get_rostime().to_nsec()*NANOS_TO_MILLIS)

        l_ang = list(left_arm.joint_angles().values())
        l_vel = list(left_arm.joint_velocities().values())
        l_eff = list(left_arm.joint_efforts().values())
        l_grip_pos = str(left_gripper.position())

        bax_row = l_ang + l_vel + l_eff
        bax_row.append(l_grip_pos)
        bax_row.append(str(t))

        with open(bax_file_name + str(sequence_counter), 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(bax_row)
        writeFile.close()


        if (watch_rows[1]==1):
            with open(watch_file_name + str(sequence_counter), 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(watch_rows[0])
            writeFile.close()
            watch_rows[1]=0

        # s to stop
        # r to remove the last file
        # n to start new sequence
        # c TWICE to shutdown the node
        shutdown = False
        if (command == 's') :
            watch_sub.unregister()
            rospy.loginfo("FINISH RECORDING SEQUENCE " + str(sequence_counter))
            rospy.loginfo("NODE STOPPED!")
            while True :
                rospy.Rate(2).sleep()

                if (command == 'r') :
                    os.remove(bax_file_name + str(sequence_counter))
                    os.remove(watch_file_name + str(sequence_counter))
                    sequence_counter = sequence_counter - 1
                    rospy.loginfo("FILE REMOVED!")
                    command = ''

                if (command == 'n') :
                    rospy.loginfo("RESET NODE!")
                    sequence_counter = sequence_counter + 1
                    resetNode()
                    watch_sub = rospy.Subscriber('/imu_giver', ImuPackage, watchCallback)
                    rospy.loginfo("START RECORDING SEQUENCE " + str(sequence_counter))
                    break

                if (command == 'c') :
                    rospy.loginfo("Enter 'c' to shutdown... ")
                    shutdown = True
                    break

        if (shutdown) :
            rospy.signal_shutdown("reason...")

    getkey_thread.join()

if __name__ == '__main__':
    listener()

