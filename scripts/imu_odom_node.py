#!/usr/bin/env python

import rospy
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import Imu
# from cv_bridge import CvBridge
# import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
# from scipy.spatial import Delaunay, delaunay_plot_2d
import io

import gtsam
from gtsam import (ISAM2, BetweenFactorConstantBias, Cal3_S2,
                   ConstantTwistScenario, ImuFactor, NonlinearFactorGraph,
                   PinholeCameraCal3_S2, Point3, Pose3,
                   PriorFactorConstantBias, PriorFactorPose3,
                   PriorFactorVector, Rot3, Values)
from gtsam.symbol_shorthand import B, V, X
from gtsam.utils import plot

import tf
import tf2_ros
import tf2_geometry_msgs  # for transforming geometry_msgs
from geometry_msgs.msg import TransformStamped


def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)

g = 9.81
n_gravity = vector3(0, 0, -g)

def preintegration_parameters():
    # IMU preintegration parameters
    # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
    PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
    I = np.eye(3)
    PARAMS.setAccelerometerCovariance(I * 0.1)
    PARAMS.setGyroscopeCovariance(I * 0.1)
    PARAMS.setIntegrationCovariance(I * 0.1)
    PARAMS.setUse2ndOrderCoriolis(False)
    PARAMS.setOmegaCoriolis(vector3(0, 0, 0))

    BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)
    DELTA = Pose3(Rot3.Rodrigues(0, 0, 0),
                  Point3(0.05, -0.10, 0.20))

    return PARAMS, BIAS_COVARIANCE, DELTA

class ImuOdomNode:
    def __init__(self):
        self.marker_pub = rospy.Publisher('/imu_odom', Marker, queue_size=10)
        self.sub = rospy.Subscriber('/robot1/imu_raw', Imu, self.imu_callback, queue_size=10000)

        self.laststamp = None

        self.n_frames_sec = 0
        self.countsec = 0

        # self.tf_publisher = tf2_ros.TransformBroadcaster()
        self.tf_broadcaster = tf.TransformBroadcaster()

        # GTSAM STUFF FROM IMUFACTORISAM2 EXAMPLE
        pose_0 = Pose3()
        print(pose_0) #EDIT - ORIGIN POSE AS START
        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = preintegration_parameters()

        # Create a factor graph
        self.graph = NonlinearFactorGraph()

        # Create (incremental) ISAM2 solver
        self.isam = ISAM2()

        # Create the initial estimate to the solution
        # Intentionally initialize the variables off from the ground truth
        self.initialEstimate = Values()
        print("INIT ESTIM:")
        print(self.initialEstimate)

        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        self.noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        self.graph.push_back(PriorFactorPose3(X(0), pose_0, self.noise))

        # Add imu priors
        self.biasKey = B(0)
        self.biasnoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.biasprior = PriorFactorConstantBias(self.biasKey, gtsam.imuBias.ConstantBias(),
                                            self.biasnoise)
        self.graph.push_back(self.biasprior)
        self.initialEstimate.insert(self.biasKey, gtsam.imuBias.ConstantBias())
        self.velnoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        # Calculate with correct initial velocity
        self.last_velocity = vector3(0, 0, 0) # EDIT - ZERO VEL PRIOR
        self.last_pose = Pose3() # EDIT - ZERO VEL PRIOR
        self.velprior = PriorFactorVector(V(0), self.last_velocity, self.velnoise)
        self.graph.push_back(self.velprior)
        self.initialEstimate.insert(V(0), self.last_velocity)

        self.accum = gtsam.PreintegratedImuMeasurements(self.PARAMS)

        self.n_msgs = 0


    def imu_callback(self, msg):
        stamp = msg.header.stamp
        if self.laststamp == None:
            self.laststamp = stamp
            return
        delta_t = stamp.to_sec() - self.laststamp.to_sec()
        # print("timestamp dif: " + str(delta_t))
        if(delta_t  < 0 ):
            print("ERROR! timestamp dif is negative!")
            return

        self.laststamp = stamp
        comp_start_time = time.time()

        self.n_msgs += 1

        # print(msg)
        # return

        # -GTSAM COMPUTATIONS-

        # t = i * delta_t  # simulation time
        t = stamp.to_sec()

        i = self.n_msgs - 1
        print("ITER: " + str(i))

        pose_0 = Pose3()
        if i == 0:  # First time add two poses
            self.initialEstimate.insert(X(0), pose_0.compose(self.DELTA))
            self.initialEstimate.insert(X(1), pose_0.compose(self.DELTA))

        elif i >= 2:  # Add more poses as necessary
            # self.initialEstimate.insert(X(i), X(i-1))
            self.initialEstimate.insert(X(i), self.last_pose)

        if i > 0:
            # Add Bias variables periodically
            if i % 5 == 0:
                self.biasKey += 1
                factor = BetweenFactorConstantBias(
                    self.biasKey - 1, self.biasKey, gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE)
                self.graph.add(factor)
                self.initialEstimate.insert(self.biasKey, gtsam.imuBias.ConstantBias())

            measuredAcc =  np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            measuredOmega =  np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

            # measuredAcc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.z, msg.linear_acceleration.y])
            # measuredOmega = np.array([msg.angular_velocity.x, msg.angular_velocity.z, msg.angular_velocity.y])

            # measuredAcc = np.array([0, 0, g])
            print("ACCELERATION: ")
            print(measuredAcc)

            self.accum.integrateMeasurement(measuredAcc, measuredOmega, delta_t)

            # Add Imu Factor
            imufac = ImuFactor(X(i - 1), V(i - 1), X(i), V(i), self.biasKey, self.accum)
            self.graph.add(imufac)

            # insert new velocity, which is wrong
            self.initialEstimate.insert(V(i), self.last_velocity)
            self.accum.resetIntegration()

        # Incremental solution
        self.isam.update(self.graph, self.initialEstimate)
        result = self.isam.calculateEstimate()
        self.last_velocity = result.atVector(V(i))
        self.last_pose = result.atPose3(X(i))

        print("RESULT: ")
        # print(result)
        print("VELOCITY:")
        print(result.atVector(V(i)))
        print("POSE:")
        print(result.atPose3(X(i)))

        comp_time = time.time() - comp_start_time
        print("computation time: " + str((comp_time) * 1000) +  " ms")
        # plot.plot_incremental_trajectory(0, result,
        #                                  start=i, scale=3, time_interval=0.01)

        # reset
        self.graph = NonlinearFactorGraph()
        self.initialEstimate.clear()

        # send tf
        self.send_gtsam_pose_as_tf(result.atPose3(X(i)), "mission_origin", "imu_odom")

    def send_gtsam_pose_as_tf(self, pose, origin_frame, end_frame):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = origin_frame
        tf_msg.child_frame_id = end_frame

        # Set the translation 
        tf_msg.transform.translation.x = pose.x()
        tf_msg.transform.translation.y = pose.y()
        tf_msg.transform.translation.z = pose.z()

        # Set the rotation 
        quat = pose.rotation().toQuaternion()
        tf_msg.transform.rotation.x = quat.x()
        tf_msg.transform.rotation.y = quat.y()
        tf_msg.transform.rotation.z = quat.z()
        tf_msg.transform.rotation.w = quat.w()


        # Broadcast the TF transform
        self.tf_broadcaster.sendTransformMessage(tf_msg)


if __name__ == '__main__':
    rospy.init_node('imu_odom_node')
    optical_flow_node = ImuOdomNode()
    rospy.spin()
