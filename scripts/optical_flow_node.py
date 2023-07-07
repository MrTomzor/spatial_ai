#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class OpticalFlowNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.pub = rospy.Publisher('optical_flow', Image, queue_size=1)
        self.sub = rospy.Subscriber('/robot1/camera1/raw', Image, self.image_callback, queue_size=10000)

        self.laststamp = None
        # cv2.startWindowThread()
        # cv2.namedWindow("preview")
        # cv2.imshow("preview", img)

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        print("w: " +str(w) + "h: " + str(h))
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

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
        # print(msg.header)
        # except CvBridgeError as e:
        #     rospy.logerr(e)
        #     return

        current_time = rospy.get_rostime()

        if self.prev_image is not None:
            dt = (current_time - self.prev_time).to_sec()
            prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) * (0.1 / tdif)
            comp_end_time = rospy.get_rostime()

            print("computation time: " + str((comp_end_time - current_time).to_sec()))

            # Visualize optical flow
            cv2.imshow('preview', self.draw_flow(current_gray, flow))
            cv2.waitKey()
            flow_vis = self.visualize_optical_flow(flow)

            try:
                self.pub.publish(self.bridge.cv2_to_imgmsg(flow_vis, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(e)



        self.prev_image = current_image
        self.prev_time = current_time

    def visualize_optical_flow(self, flow):
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return flow_vis

if __name__ == '__main__':
    rospy.init_node('optical_flow_node')
    optical_flow_node = OpticalFlowNode()
    rospy.spin()
    cv2.destroyAllWindows()
