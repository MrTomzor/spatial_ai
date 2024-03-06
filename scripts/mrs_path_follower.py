import copy
import rospy
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
import threading

import heapq

import rospkg

from spatial_ai.common_spatial import *
from spatial_ai.fire_slam_module import *
from spatial_ai.submap_builder_module import *
from spatial_ai.local_navigator_module import *

from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

import trimesh

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
import io
import copy

from scipy.spatial.transform import Rotation as R
import tf
import tf2_ros
import tf2_geometry_msgs  # for transforming geometry_msgs
import tf.transformations as tfs
from geometry_msgs.msg import TransformStamped


import matplotlib.pyplot as plt
import numpy as np
import sys

from geometry_msgs.msg import Twist

class FollowerNode():
    def __init__(self):# # #{
        odom_topic = '/ov_msckf/odomimu'
        path_topic = '/uav1/trajectory_generation/path'
        twist_topic = '/twist_velocity_control'
        self.imu_frame = 'imu'
        self.fcu_frame = 'imu'
        self.camera_frame = 'cam0'
        self.odom_frame = 'global'
        self.tf_listener = tf.TransformListener()

        self.path_mutex = threading.Lock()
        self.has_path = False
        self.path_index = 0
        self.path_points_global = None
        self.path_headings_global = None

        self.reaching_dist = 0.5
        self.reaching_heading = np.pi / 8

        # PUB
        self.pub_twist = rospy.Publisher(twist_topic, Twist, queue_size=10)

        # SUB
        self.sub_path = rospy.Subscriber(path_topic, mrs_msgs.msg.Path, self.path_msg_callback, queue_size=10)

        # TIMER
        self.control_rate = 10
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_iter)
    # # #}

    def send_twist_command(self, vel, angvel):# # #{
        twist_msg = Twist()

        # Set linear velocity (m/s)
        twist_msg.linear.x = vel[0]  
        twist_msg.linear.y = vel[1]  
        twist_msg.linear.z = vel[2]  

        # Set angular velocity (rad/s)
        twist_msg.angular.x = angvel[0]
        twist_msg.angular.y = angvel[1]
        twist_msg.angular.z = angvel[2]
        # print("SENDING TWIST:")
        # print(vel)
        # print(angvel)

        self.pub_twist.publish(twist_msg)
    # # #}

    def get_fake_path(self):
        T_odom = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
        cur_pos = T_odom[:3,3]
        cur_h = transformationMatrixToHeading(T_odom)

        pts_global = cur_pos.reshape((1,3))
        h_global = np.array([cur_h])

        self.has_path = True
        self.path_index = 0
        self.path_len = pts_global.shape[0]
        self.path_points_global = pts_global
        self.path_headings_global = h_global

    def control_iter(self, event=None):# # #{
        with ScopedLock(self.path_mutex):
            if not self.has_path:
                # BRAKE
                # TODO - send zero vel command
                # self.get_fake_path()
                return

            T_odom = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
            cur_pos = T_odom[:3,3]
            cur_h = transformationMatrixToHeading(T_odom)

            # GET NEXT VP TO REACH
            self.path_index
            next_pos = self.path_points_global[self.path_index]
            next_h = self.path_headings_global[self.path_index]

            pos_error = (next_pos - cur_pos )
            distance_error = np.linalg.norm(pos_error)
            heading_error = np.unwrap(np.array([next_h]) - np.array([cur_h]))[0]
            # print("DIST ERROR: " + str(distance_error) + " HEADING ERR: " +str(heading_error))

            if distance_error  < self.reaching_dist and np.abs(heading_error) < self.reaching_heading:
                print("PROGRESS: " + str(self.path_index) + "/" + str(self.path_len))
                print("REACHED NEXT PT!")
                self.path_index += 1
                if self.path_index == self.path_len:
                    print("PATH END REACHED")
                    self.has_path = False
                    self.send_twist_command(np.zeros((3,1)), np.zeros((3,1)))
                    return

            # COMPUTE VEL AND ANGVEL TO REACH THE DESIRED POSE
            rot_target_matrix = R.from_euler('z', next_h, degrees=False).as_matrix()
            rot_dif_matrix = np.linalg.inv(T_odom[:3, :3]) @ rot_target_matrix

            rot_error_vec = R.from_matrix(rot_dif_matrix).as_rotvec() 
            # print("ROT DIF ROTVEC: ")
            # print(rot_error_vec )

            p_linear = 2.0
            p_angular = 0.8
            # max_vel = 5

            vel_cmd = (pos_error) * p_linear
            angvel_cmd = (rot_error_vec ) * p_angular

            vel_cmd = np.linalg.inv(T_odom[:3, :3]) @ vel_cmd

            # CONSTRUCT MSG AND SEND
            self.send_twist_command(vel_cmd, angvel_cmd)
    # # #}

    def path_msg_callback(self, msg):# # #{
        print("NEW PATH!")

        with ScopedLock(self.path_mutex):
            # READ PTS AND HEADINGS
            pts_array = []
            h_array = []
            for i in range(len(msg.points)):
                pts_array.append([msg.points[i].position.x,msg.points[i].position.y,msg.points[i].position.z])
                h_array.append(msg.points[i].heading)
            pts_array = np.array(pts_array)
            h_array = np.array(h_array)

            # TRANSFORM THEM TO ODOM FRAME
            T_msg_to_global = lookupTransformAsMatrix(msg.header.frame_id, self.odom_frame, self.tf_listener)
            pts_global, h_global = transformViewpoints(pts_array, h_array, np.linalg.inv(T_msg_to_global))

            self.has_path = True
            self.path_index = 0
            self.path_len = pts_global.shape[0]
            self.path_points_global = pts_global
            self.path_headings_global = h_global
# # #}



if __name__ == '__main__':
    rospy.init_node('mrs_path_pid_follower')
    optical_flow_node = FollowerNode()
    rospy.spin()
    cv2.destroyAllWindows()
