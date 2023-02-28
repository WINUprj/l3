#!/usr/bin/env python3
"""
lane_following.py

Custom functionalities required for lane following.

This file includes:
    - LaneFollowing class (equivalent to nodes)
    - 

Pipeline note:
1. color detection
2. detect line segments
3. extract pose estimate from each segment 
4. run PID controller

TODO:
    - Check the histogram with white curve
    - get average center of contours
    - If histogram does not work that much, use angle estimation.
"""
import time
import yaml

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
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
                 controller_flags=[True, True, True],
                 init_vleft=0.5,
                 init_vright=0.5):
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

        # PID related parameters
        assert sum(controller_flags) > 0, "At least one of P, I, D is needed."
        self._controller_flags = controller_flags
        self.P = 0.1
        self.I = 0.01
        self.D = 0.01
        self.prev_integ = 0.
        self.prev_err = 0.
        self.M = 0.

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
            f"/{self._veh}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1
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
    
    def estimate_lane_median(self, cont):
        center_x = []   # x coordinate of contours' center
        for c in cont:
            mom = cv2.moments(c)    # Get momentum of a contour
            if mom["m00"] == 0 or cv2.contourArea(c) < 900:
                # If denomenator of momentum is 0 or area of contours are
                # mostly equivalent to noise, ignore
                continue
            x = int(mom["m10"] / mom["m00"])
            center_x.append(x)
        
        if len(center_x == 0):
            return -1
        else:
            return int(np.median(center_x))

    def validate_line_coordinates(self, yx, wx):
        if wx == -1 and yx == -1:
            # Both lines are lost
            wx, yx = None, None
        elif wx == -1 and yx >= 0:
            # Yellow line is visible but white line is lost
            wx = yx - (self.is_eng * self._default_width)
        elif wx >= 0 and yx == -1:
            # White line is visible but yellow line is lost
            yx = wx + (self.is_eng * self._default_width)
        elif self.is_eng == 1 and wx > yx:
            # For English lane following, white lane must be at left
            # So with wx > yx, we are detecting line beyond the yellow line
            wx = yx - self._default_width
        elif self.is_eng == -1 and wx < yx:
            yx = wx - self._default_width

        return yx, wx
            
    def pid(self, cur_val, target=0):
        # Compute error
        err = cur_val - target
        print(f"Error: {err}")
        prop_term = 0.
        integ_term = 0.
        deriv_term = 0.

        # Compute P term
        if self._controller_flags[0]:
            prop_term = self.P * err
        
        # Compute I term
        # Here we assume _update_frequency as dt
        if self._controller_flags[1]:
            integ_term = self.I * (self.prev_integ + (err * self._update_freq))
            self.prev_integ = integ_term
        
        # Compute D term
        if self._controller_flags[2]:
            deriv_term = self.D * ((err - self.prev_err) /  self._update_freq)
            self.prev_err = err

        self.M = prop_term + integ_term + deriv_term
    
    def pid_wrapper(self, yellow_mask, white_mask):
        # Mask out the 40% of upper region in the masks. As a result, we could
        # avoid the noise which are not related to lane
        yellow_mask[:int(0.4*self.height), :] = 0
        white_mask[:int(0.4*self.height), :] = 0

        # Warp masks with pre-calculated custom homography matrix
        yellow_transformed = cv2.warpPerspective(yellow_mask,
                                                 self.hom_mat,
                                                 (self.width, self.height))
        white_transformed = cv2.warpPerspective(white_mask,
                                                self.hom_mat,
                                                (self.width, self.height))
        
        # Compute the contours
        yellow_cont, _ = cv2.findContours(yellow_transformed,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        white_cont, _ = cv2.findContours(white_transformed,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        
        # Estimate the median of where each lines are located
        yx = self.estimate_lane_median(yellow_cont)
        wx = self.estimate_lane_median(white_cont)

        # Check the validity of yx and wx values, and auto-adjust if they are not
        yx, wx = self.validate_line_coordinates(yx, wx)
        
        if yx == None and wx == None:
            raise Exception("Duckiebot lost the lines from its vision.")

        # Estimate the middle point relative to middle of camera view
        med = (yx + wx) // 2
        med -= (self.width // 2)
        med /= (self.width)
        
        # Call the pid method to run the controller
        self.pid(med, 0)
        
        # Adjust the current velocity and publish
        print(f"Yellow med: {yx}, White med: {wx}")
        print(f"current M: {self.M}")
        self.cur_vleft += self.M
        self.cur_vright -= self.M
        print(f"Current vleft: {self.cur_vleft}, vright: {self.cur_vright}")
        self.publish_maneuver(self.cur_vleft, self.cur_vright)

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
            white_mask = cv2.inRange(hsv_img, self._white_low, self._white_high)
            
            self.pid_wrapper(yellow_mask, white_mask)

            yellow_contour, _ = cv2.findContours(yellow_mask,
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
            white_contour, _ = cv2.findContours(white_mask,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

            img = cv2.drawContours(img, yellow_contour, -1, (0, 0, 255), 2)
            img = cv2.drawContours(img, white_contour, -1, (0, 255, 0), 2)

            img_msg = self.bridge.cv2_to_compressed_imgmsg(img)
            self.pub_cam.publish(img_msg)

            self.freq_count = 0
        
        self.freq_count += 1
    
    # def drive(self, vleft, vright):
    #     ### Only call when it starts and when it ends
    #     self.cur_vleft = vleft
    #     self.cur_vright = vright
    #     self.publish_maneuver(vleft, vright)

    def shutdown_hook(self):
        self.cur_vleft = 0
        self.cur_vright = 0
        self.publish_maneuver(self.cur_vleft, self.cur_vright)

if __name__ == "__main__":
    lane_following = LaneFollow()
    # start = time.time()
    rospy.on_shutdown(lane_following.shutdown_hook)
    # while time.time() - start <= 6:
    rospy.spin()
