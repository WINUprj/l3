#!/usr/bin/env python3
"""
lane_following.py

Custom functionalities required for lane following.

This file includes:
    - LaneFollowing class (equivalent to nodes)

Algorithm:

References
----------
https://medium.com/@SunEdition/lane-detection-and-turn-prediction-algorithm-for-autonomous-vehicles-6423f77dc841
"""
import time
import yaml

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2
from cv_bridge import CvBridge


def read_yaml_file(path):
    with open(path, 'r') as f:
        try:
            yaml_dict = yaml.safe_load(f)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(f"YAML syntax error. File: {path}. Exc: {exc}")
            rospy.signal_shutdown()
            return


def parse_calib_params(int_path=None, ext_path=None):
    # Load dictionaries from files
    int_dict, ext_dict = None, None
    if int_path:
        int_dict = read_yaml_file(int_path)
    if ext_path:
        ext_dict = read_yaml_file(ext_path)
    
    # Reconstruct the matrices from loaded dictionaries
    camera_mat, distort_coef, proj_mat = None, None, None
    hom_mat = None
    if int_dict:
        camera_mat = np.array(list(map(np.float32, int_dict["camera_matrix"]["data"]))).reshape((3, 3))
        distort_coef = np.array(list(map(np.float32, int_dict["distortion_coefficients"]["data"]))).reshape((1, 5))
        proj_mat = np.array(list(map(np.float32, int_dict["projection_matrix"]["data"]))).reshape((3, 4))
    if ext_dict:
        hom_mat = np.array(list(map(np.float32, ext_dict["homography"]))).reshape((3, 3))

    return (camera_mat, distort_coef, proj_mat, hom_mat)


class LaneFollow(DTROS):
    def __init__(self,
                 node_name="lane_follow",
                 update_freq=2,
                 is_eng=1,
                 init_vel=0.25):
        super(LaneFollow, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )

        self._update_freq = update_freq
        self._veh = rospy.get_param("~veh")
        self._int_path = rospy.get_param("~int_file")
        # Yellow range: https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv
        self._yellow_low = np.array([20, 50, 50])
        self._yellow_high = np.array([30, 255, 255])

        # Callback management parameters 
        self.freq_count = 0

        # Comupter vision related parameters
        self.width = None
        self.height = None
        self.is_eng = is_eng
        self.bridge = CvBridge()
        self.camera_mat, self.dist_coef, _, _ = parse_calib_params(int_path=self._int_path)

        # PID related parameters
        self.P = 0.035
        self.I = 0.0
        self.D = 0.0
        self.prev_integ = 0.
        self.prev_err = 0.
        self.M = 0.
        self.offset = 0
        self.target = 0
        self.var = 40
        self.pv = []

        # Maneuvering parameters
        self.cur_vel = init_vel
        self.cur_ang = 0.
        
        # Subscriber
        self.sub_cam = rospy.Subscriber(
            f"/{self._veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_col_detect
        )

        # Publishers
        self.pub_cam = rospy.Publisher(
            f"/{self._veh}/{node_name}/colored/compressed",
            CompressedImage,
            queue_size=1
        )
        self.pub_man = rospy.Publisher(
            f"/{self._veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

    def publish_maneuver(self, vel, ang):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = vel
        msg.omega = ang
        self.pub_man.publish(msg)

    def undistort(self, img):
        return cv2.undistort(img,
                             self.camera_mat,
                             self.dist_coef,
                             None,
                             self.camera_mat)
    
    def get_med(self, cont):
        med = 0
        mx_area = -1
        for c in cont:
            area = cv2.contourArea(c)
            if area > mx_area:
                mom = cv2.moments(c)    # Get momentum of a contour
                if mom["m00"] == 0.0:
                    # If denomenator of momentum is 0
                    continue
                med = int(mom["m10"] / mom["m00"])
                mx_area = area

        return med

    def pid(self, cur_val, target):
        # Compute error
        err = cur_val - target
        if abs(err) < self.var:
            err = 0
        print(f"Error: {err}")
        prop_term = 0.
        integ_term = 0.
        deriv_term = 0.

        # Compute P term
        prop_term = self.P * err
        
        # Compute I term
        # Here we assume _update_frequency as dt
        integ_term = self.I * (self.prev_integ + (err * self._update_freq))
        self.prev_integ = integ_term
        
        # Compute D term
        deriv_term = self.D * ((err - self.prev_err) /  self._update_freq)
        self.prev_err = err

        self.M = prop_term + integ_term + deriv_term
    
    #TODO: Refactor to use twisted 2D
    def pid_wrapper(self, img, yellow_mask):
        # Mask out the 40% of upper region in the masks. As a result, we could
        # avoid the noise which are not related to lane
        yellow_mask[:int(0.4*self.height), :] = 0

        # Compute the contours
        yellow_cont, _ = cv2.findContours(yellow_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        
        # Estimate the median of where each lines are located
        yx = self.get_med(yellow_cont)

        self.pv.append(yx)
        yx = int(np.mean(self.pv[max(-10, -len(self.pv)):]))
        self.pid(yx + self.offset, self.width//2)
        
        # Adjust the current velocity and publish
        self.cur_ang = -self.M
        self.cur_ang = np.clip(self.cur_ang, -360, 360)

        self.publish_maneuver(self.cur_vel, self.cur_ang)

    def cb_col_detect(self, msg):
        if self.freq_count % self._update_freq == 0:
            img = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            img = self.undistort(img)
            
            if self.width is None:
                self.width = img.shape[1]
            if self.height is None:
                self.height = img.shape[0]

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            yellow_mask = cv2.inRange(hsv_img, self._yellow_low, self._yellow_high)

            self.pid_wrapper(img, yellow_mask)

            self.freq_count = 0
        
        self.freq_count += 1

    def shutdown_hook(self):
        self.publish_maneuver(0, 0)

if __name__ == "__main__":
    lane_following = LaneFollow()
    rospy.on_shutdown(lane_following.shutdown_hook)
    rospy.spin()
