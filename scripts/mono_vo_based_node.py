#!/usr/bin/env python

import rospy
# from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker
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
from geometry_msgs.msg import TransformStamped


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 200
kMaxNumFeature = 2000

# LKPARAMS
lk_params = dict(winSize  = (31, 31),
                 #maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2

class OdomNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None

        self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)
        self.sub = rospy.Subscriber('/robot1/camera1/raw', Image, self.image_callback, queue_size=10000)

        self.tf_broadcaster = tf.TransformBroadcaster()
        # self.sub = rospy.Subscriber('/robot1/camera1/compressed', CompressedImage, self.image_callback, queue_size=10000)

        self.laststamp = None
        self.orb = cv2.ORB_create(nfeatures=3000)

        # Load calib
        self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
        # self.K = np.array([, 644.5958939934509, 400.0503960299562, 300.5824096896595]).reshape((3,3))

        # self.K = np.array([642.8495341420769, 644.5958939934509, 400.0503960299562, 300.5824096896595]).reshape((3,3))
        self.P = np.zeros((3,4))
        self.P[:3, :3] = self.K

        # NEW
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        # self.focal = cam.fx
        # self.pp = (cam.cx, cam.cy)
        self.focal = 643.3520818301457
        self.pp = (400, 300)
        self.width = 800
        self.height = 600
        self.tracking_bin_width = 50
        self.min_features_per_bin = 2
        
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)

        self.tracking_colors = np.random.randint(0, 255, (100, 3)) 

        self.n_frames = 0

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        stamp = msg.header.stamp

        comp_start_time = rospy.get_rostime()

        # convert img to grayscale and shift buffer
        self.last_frame = self.new_frame
        self.new_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print("FRAME " + str(self.n_frames))

        comp_start_time = time.time()

        if self.n_frames == 0:
            # FIND FIRST FEATURES
            self.px_ref = self.detector.detect(self.new_frame)
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
            print("FOUND FEATURES: " + str(self.px_ref.shape[0]))
            self.n_frames = 1
            return

        # TRACK
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)

        '''
        # FIND ESSENTIAL MATRIX
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)

        # INIT ROTATION AND TRANSLATION AT 2ND FRMAE
        if self.n_frames == 1:
            self.cur_R = R
            self.cur_t = t
        else:
            absolute_scale = 1
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        print("T:")
        print(self.cur_t)
        # print(self.cur_t)
        '''

        # FIND FEATS IF NOT ENOUGH!
        if(self.px_ref.shape[0] < kMinNumFeature):
            print("FINDING FEATURES")
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

        # FIND SPARSE AREAS WITH HISTOGRAM!
        wbins = self.width // self.tracking_bin_width
        hbins = self.height // self.tracking_bin_width
        trackhist = np.histogram2d(self.px_cur[:, 0], self.px_cur[:, 1], [wbins, hbins])
        print(trackhist[0])
        print(trackhist[0].shape)
        bins_to_regenerate = trackhist[0] < self.min_features_per_bin

        print(bins_to_regenerate)
        print("REGENERATING IN " + str(np.sum(bins_to_regenerate, None)) + " BINS")
        found_total = 0
        for xx in range(wbins):
            for yy in range(hbins):
                bin_coords = [xx * self.tracking_bin_width, (xx+1) * self.tracking_bin_width, yy * self.tracking_bin_width, (yy+1) * self.tracking_bin_width]
                locally_found = self.detector.detect(self.new_frame[bin_coords [2] : bin_coords [3], bin_coords [0] : bin_coords [1]])
                found_total += len(locally_found)
                if len(locally_found) == 0:
                    continue

                locally_found = np.array([x.pt for x in locally_found], dtype=np.float32)
                locally_found[:, 0] += bin_coords[0]
                locally_found[:, 1] += bin_coords[2]
                self.px_cur = np.concatenate((self.px_cur, locally_found))

        print("FOUND IN BINS: " + str(found_total))

        print("TRACKED FEATURES: " + str(self.px_ref.shape[0]))


        # VISUALIZE FEATURES
        self.n_frames += 1

        vis = self.visualize_tracking()
        self.kp_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        # self.publish_pose_msg()

        comp_time = time.time() - comp_start_time
        print("computation time: " + str((comp_time) * 1000) +  " ms")


    def visualize_tracking(self):
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        ll = np.array([0, 0])  # lower-left
        ur = np.array([self.width, self.height])  # upper-right
        inidx = np.all(np.logical_and(ll <= self.px_cur, self.px_cur <= ur), axis=1)
        inside_pix_idxs = self.px_cur[inidx].astype(int)

        rgb[inside_pix_idxs[:, 1],inside_pix_idxs[:, 0], 0] = 255
        for i in range(inside_pix_idxs.shape[0]):
            rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), 5, 
                           (255, 0, 0), -1) 

        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis
    
    def publish_pose_msg(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "mission_origin"
        tf_msg.child_frame_id = "cam_odom"

        # Set the translation 
        tf_msg.transform.translation.x = self.cur_t[0]
        tf_msg.transform.translation.y = self.cur_t[1]
        tf_msg.transform.translation.z = self.cur_t[2]

        # Set the rotation 
        r = Rotation.from_matrix(self.cur_R)
        quat = r.as_quat()
        # quat = pose.rotation().toQuaternion()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]


        # Broadcast the TF transform
        self.tf_broadcaster.sendTransformMessage(tf_msg)

    def visualize_keypoints(self, img, kp):
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for k in kp:
            rgb[int(k.pt[1]), int(k.pt[0]), 0] = 255
        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis

    def visualize_keypoints_in_space(self, kp):
        point_cloud = PointCloud()
        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = 'mission_origin'  # Set the frame ID according to your robot's configuration

        # Sample points
        for i in range(kp.shape[1]):
            if kp[2, i] > 0:
                point1 = Point32()
                point1.x = kp[0, i]
                point1.y = kp[1, i]
                point1.z = kp[2, i]
                point_cloud.points.append(point1)

        self.kp_pcl_pub.publish(point_cloud)

if __name__ == '__main__':
    rospy.init_node('visual_odom_node')
    optical_flow_node = OdomNode()
    rospy.spin()
    cv2.destroyAllWindows()
