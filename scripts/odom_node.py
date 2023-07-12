#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class OdomNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None

        self.pub = rospy.Publisher('optical_flow', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/tracked_features', Marker, queue_size=10)
        self.sub = rospy.Subscriber('/robot1/camera1/raw', Image, self.image_callback, queue_size=10000)

        self.laststamp = None
        self.orb = cv2.ORB_create(3000)

        # Load calib
        self.K = np.array([415, 0, 320, 0, 415, 240, 0, 0, 1]).reshape((3,3))
        self.P = np.zeros((3,4))
        self.P[:3, :3] = self.K

        self.prev_kp = None
        self.prev_desc = None
        self.current_pose = np.eye(4)

        # FLANN
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        # cv2.startWindowThread()
        # cv2.namedWindow("preview")
        # cv2.imshow("preview", img)

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


    def image_callback(self, msg):
        # try:
        current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        stamp = msg.header.stamp
        tdif = 0
        if self.laststamp != None:
            tdif = stamp.to_sec() - self.laststamp.to_sec()
            print("timestamp dif: " + str(tdif))
            if(tdif < 0 ):
                print("ERROR! timestamp dif is negative!")
                return
        self.laststamp = stamp
        comp_start_time = rospy.get_rostime()

        # convert img to grayscale
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

        # detect and describe keypoints
        kp, desc = self.orb.detectAndCompute(current_gray, None)

        # if first frame, return
        if self.prev_kp == None:
            self.prev_kp = kp
            self.prev_desc = desc
            return

        comp_end_time = rospy.get_rostime()

        # Find matches
        matches = self.flann.knnMatch(self.prev_desc, desc, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # Get the image points form the good matches
        q1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        q2 = np.float32([kp[m.trainIdx].pt for m in good])

        # get pose
        transf = self.get_pose(q1, q2)

        print("total computation time: " + str((comp_end_time - comp_start_time).to_sec()))
        print("good matches: " + str(len(good))+ "/" + str(len(matches)))
        print("transform: ")
        print(transf)
        if(np.all(np.isfinite(transf))):
            self.current_pose = np.matmul(self.current_pose, np.linalg.inv(transf))
        else:
            print("NOT FINITE TRANSFORM!!!")

        print("current pose: ")
        print(self.current_pose)
        # Visualize orb
        # vis = self.visualize_keypoints(current_gray, kp)
        # self.pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

        self.prev_kp = kp
        self.prev_desc = desc

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

    def visualize_optical_flow(self, flow):
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return flow_vis

    def visualize_corners(self, corners):
        rgb = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
        thresh = 0.01 * corners.max()
        rgb[corners > thresh, 0] = 255

        # for i in range(corners.shape[0]):
        #     for j in range(corners.shape[1]):
        #         if corners[i, j] > thresh:
        #             rgb[i,j,0] = 255
        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis

    def visualize_keypoints(self, img, kp):
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # thresh = 0.01 * corners.max()
        # rgb[corners > thresh, 0] = 255
        # coords = np.zeros((len(kp),2))
        # for i in range(len(kp)):
        #     # print(k)
        #     # rgb[corners > thresh, 0] = 255
        #     coords[i,0] = ()
        #     coords[i,1] = ()
        for k in kp:
            rgb[int(k.pt[1]), int(k.pt[0]), 0] = 255

        # for i in range(corners.shape[0]):
        #     for j in range(corners.shape[1]):
        #         if corners[i, j] > thresh:
        #             rgb[i,j,0] = 255
        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis

if __name__ == '__main__':
    rospy.init_node('optical_flow_node')
    optical_flow_node = OdomNode()
    rospy.spin()
    cv2.destroyAllWindows()
