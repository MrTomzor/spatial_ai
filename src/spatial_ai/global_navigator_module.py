
# #{ imports

import copy
import rospy
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
import threading

import heapq

import dbow
import rospkg

from spatial_ai.common_spatial import *

from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import mrs_msgs.msg
import std_msgs.msg
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import inspect
from shapely import geometry

import trimesh
import rtree

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

from scipy.spatial.transform import Rotation
import tf
import tf2_ros
import tf2_geometry_msgs  # for transforming geometry_msgs
import tf.transformations as tfs
from geometry_msgs.msg import TransformStamped


import gtsam
from gtsam import DoglegOptimizer
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import L, X
from gtsam.examples import SFMdata
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import sys
from termcolor import colored, cprint

from scipy.spatial.transform import Rotation as R

# #}

class GlobalNavigatorModule:
    def __init__(self, mapper, local_navigator, ptraj_topic, output_path_topic):# # #{

        self.mapper = mapper
        self.local_navigator = local_navigator
        self.odom_frame = mapper.odom_frame
        self.fcu_frame = mapper.fcu_frame
        self.tf_listener = mapper.tf_listener

        # VIS PUB
        # self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        # self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        self.marker_scale = 0.15
        self.path_step_size = 0.5
        # self.max_heading_change_per_m = np.pi / 10
        self.max_heading_change_per_m = np.pi / 6

        self.safety_replanning_trigger_odist = 0.2
        self.min_planning_odist = 0.2
        self.max_planning_odist = 2

        # # PREDICTED TRAJ
        # self.sub_predicted_trajectory = rospy.Subscriber(ptraj_topic, mrs_msgs.msg.MpcPredictionFullState, self.predicted_trajectory_callback, queue_size=10000)
        # self.predicted_trajectory_pts_global = None

        # PLANNING PARAMS
        self.global_roadmap = None
        self.global_roadmap_index = None
        self.global_roadmap_len = None
        self.roadmap_start_time = None

        self.local_nav_start_time = rospy.get_rostime()
        self.local_reaching_dist = 3
        self.last_traj_send_time =  rospy.get_rostime()
        self.traj_min_duration = 10

        self.currently_navigating_pts = None
        self.current_goal_vp_global = None
        self.reaching_dist = 1
        self.reaching_angle = np.pi/2

        self.max_goal_vp_pathfinding_times = 3
        self.current_goal_vp_pathfinding_times = 0

        self.fspace_bonus_mod = 2
        self.safety_weight = 5

        self.last_matching_time = None
        self.matching_min_interval = 1

        self.best_current_match = None
        self.best_current_match_score = None
        
        # # #}

    def main_iter(self):# # #{
        # self.dumb_forward_flight_rrt_iter()

        if True:
            cur_smap_copy = None
            adjacent_smaps = None
            mchunk_len = None
            # copied_
            with ScopedLock(self.mapper.spheremap_mutex):
                cur_smap_copy = copy.deepcopy(self.mapper.spheremap)
                mchunk_len = len(self.mapper.mchunk.submaps)
                # TODO - stop when odometry unasfe (or maybe tear that to new mchunk!)
            max_num_submaps = 3
            prev_smaps_indices = []
            # TODO ... DRAW!!!
            # TODO - find smaps coherent with the current one!!! - DRAW!! simplest = just the prev N smaps in chunk!


        return

        # TODO - try matching time!
    # # #}


