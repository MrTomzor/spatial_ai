from spatial_ai.fire_slam_module import *

if __name__ == '__main__':
    width = 960
    height = 720
    K = np.array([933.5640667549508, 0.0, 500.5657553739987, 0.0, 931.5001605952165, 379.0130687255228, 0.0, 0.0, 1.0]).reshape((3,3))
    camera_frame_id = "uav1/rgb"
    odom_orig_frame_id = "uav1/passthrough_origin"
    image_sub_topic = '/uav1/tellopy_wrapper/rgb/image_raw'

    # self.width = 800
    # self.height = 600
    # self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
    # self.camera_frame_id = "cam0"
    # self.odom_orig_frame_id = "global"
    # self.rgb_topic = '/robot1/camera1/raw'
    # self.kf_dist_thr = 4
    # self.toonear_vis_dist = 20

    rospy.init_node('fire_slam_node')
    slam_node = FireSLAMModule(width, height, K, camera_frame_id, odom_orig_frame_id, image_sub_topic, standalone_mode=True)
    rospy.spin()
    cv2.destroyAllWindows()
