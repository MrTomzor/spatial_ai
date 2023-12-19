from spatial_ai.fire_slam_module import *

if __name__ == '__main__':
    width = 960
    height = 720
    K = np.array([933.5640667549508, 0.0, 500.5657553739987, 0.0, 931.5001605952165, 379.0130687255228, 0.0, 0.0, 1.0]).reshape((3,3))
    camera_frame_id = "uav1/rgb"
    odom_orig_frame_id = "uav1/passthrough_origin"
    image_sub_topic = '/uav1/tellopy_wrapper/rgb/image_raw'

    rospy.init_node('fire_slam_node')
    slam_node = FireSLAMModule(width, height, K, camera_frame_id, odom_orig_frame_id, None, image_sub_topic, standalone_mode=True)
    rospy.spin()
    cv2.destroyAllWindows()
