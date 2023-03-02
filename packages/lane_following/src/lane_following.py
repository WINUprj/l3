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
                 update_freq=3,
                 is_eng=1,
                 init_vleft=0.3,
                 init_vright=0.3):
        super(LaneFollow, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )

        self._update_freq = update_freq
        self._veh = rospy.get_param("~veh")
        self._int_path = rospy.get_param("~int_file")
        self._homography_path = rospy.get_param("~homography_file")
        # Yellow range: https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv
        self._yellow_low = np.array([20, 50, 50])
        self._yellow_high = np.array([30, 255, 255])
        # White range: https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
        self._white_low = np.array([0, 0, 200])
        self._white_high = np.array([255, 55, 255])

        # Callback management parameters 
        self.freq_count = 0

        # Comupter vision related parameters
        self.width = None
        self.height = None
        self._thresh = 10000
        self._default_width = 450
        self.is_eng = is_eng
        self.bridge = CvBridge()
        self.camera_mat, self.dist_coef, _, _ = parse_calib_params(int_path=self._int_path)
        self.hom_mat = np.load(self._homography_path)
        self.zero = None

        # PID related parameters
        self.P = 0.00008
        self.I = 0.0
        self.D = 0.0
        self.prev_integ = 0.
        self.prev_err = 0.
        self.M = 0.
        self.offset = -210
        self.target = 0

        # Maneuvering parameters
        self.cur_vleft = init_vleft
        self.cur_vright = init_vright
        self.i = 0
        
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
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

    def publish_maneuver(self, vleft, vright):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = vleft
        msg.vel_right = vright
        
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
        # white_mask[:int(0.4*self.height), :] = 0        

        # Compute the contours
        yellow_cont, _ = cv2.findContours(yellow_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        
        # Estimate the median of where each lines are located
        yx = self.get_med(yellow_cont)

        self.pid(yx + self.offset, self.width//2)
        
        # Adjust the current velocity and publish
        print(f"Yellow med: {yx}, med: {self.zero}")
        print(f"current M: {self.M}")
        self.cur_vleft += self.M
        self.cur_vleft = np.clip(self.cur_vleft, 0.1, 0.5)
        self.cur_vright -= self.M
        self.cur_vright = np.clip(self.cur_vright, 0.1, 0.5)
        print(f"Current vleft: {self.cur_vleft}, vright: {self.cur_vright}")

        self.publish_maneuver(self.cur_vleft, self.cur_vright)

        img[:, yx] = (0, 255, 0)
        l, h = min(yx+self.offset, self.width//2), max(yx+self.offset, self.width//2)
        img[self.height//2, l:h] = (0, 0, 255) if (yx+self.offset > 0) else (255, 0, 0)
        
        img_msg = self.bridge.cv2_to_compressed_imgmsg(img)
        self.pub_cam.publish(img_msg)

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

            # img_msg = self.bridge.cv2_to_compressed_imgmsg(img)
            # self.pub_cam.publish(img_msg)

            self.freq_count = 0
        
        self.freq_count += 1

    def shutdown_hook(self):
        self.cur_vleft = 0
        self.cur_vright = 0
        self.publish_maneuver(self.cur_vleft, self.cur_vright)

if __name__ == "__main__":
    lane_following = LaneFollow()
    lane_following.publisher()
    # rospy.on_shutdown(lane_following.shutdown_hook)
    # rospy.spin()
