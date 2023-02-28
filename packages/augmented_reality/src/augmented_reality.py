#!/usr/bin/env python3
import os
import time
import yaml

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2
from cv_bridge import CvBridge


class Augmenter(DTROS):
    def __init__(self, node_name="augmented_reality"):
        super(Augmenter, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )

        # Get arguments
        self._veh = rospy.get_param("~veh")
        self._map_file = rospy.get_param("~map_file")

        self._intrinsic_path = f"/data/config/calibrations/camera_intrinsic/{self._veh}.yaml"
        self._extrinsic_path = f"/data/config/calibrations/camera_extrinsic/{self._veh}.yaml"
        self._bridge = CvBridge()

        self.camera_mat = None
        self.proj_mat = None
        self.hom_mat = None
        self.distort_coef = None
        self.map_dic = None

        self.h = None
        self.w = None

        self.cnt = 0

        # Load intrinsic and extrinsic parameters
        self.parse_calib_params()
        # self.ground2pixel()

        # Subscribers
        self.sub_cam = rospy.Subscriber(
            f"/{self._veh}/camera_node/image/compressed",
            CompressedImage,
            self.callback
        )
        
        # Publishers
        self.pub_cam = rospy.Publisher(
            f"/{self._veh}/{node_name}/image/compressed",
            CompressedImage,
            queue_size=10
        )

    def parse_calib_params(self):
        int_dic = self.read_yaml_file(self._intrinsic_path)
        ext_dic = self.read_yaml_file(self._extrinsic_path)
        
        # Extract main matrices
        self.camera_mat = np.array(list(map(np.float32, int_dic["camera_matrix"]["data"]))).reshape((3, 3))
        self.distort_coef = np.array(list(map(np.float32, int_dic["distortion_coefficients"]["data"]))).reshape((1, 5))
        self.proj_mat = np.array(list(map(np.float32, int_dic["projection_matrix"]["data"]))).reshape((3, 4))
        self.hom_mat = np.array(list(map(np.float32, ext_dic["homography"]))).reshape((3, 3))

    def ground2pixel(self):
        self.map_dic = self.read_yaml_file(self._map_file)

        # Convert points to pixel coordinates
        self.pts = {}
        half_size = np.array([self.h // 2, self.w // 2, 0])
        print(self.hom_mat)
        for name, pt in self.map_dic["points"].items():
            rf = pt[1]
            self.pts[name] = self.hom_mat.dot(rf)
            print(self.pts[name])
            self.pts[name] = (self.pts[name] * half_size) + half_size
            self.pts[name] = self.pts[name].astype(int)
        
        print(self.pts)

    def render_segments(self, img):
        for d in self.map_dic["segments"]:
            keys = d["points"]
            p = [None, None]
            p[0] = self.pts[keys[0]]
            p[1] = self.pts[keys[1]]
            col = d["color"]
            img = self.draw_segment(img, [p[0][0], p[1][0]], [p[0][1], p[1][1]], col)

        print("rendered the segments!")
        cv2.imwrite("/data/segment_rendered.png", img)
        return img 

    def read_yaml_file(self, fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.safe_load(in_file)
                return yaml_dict

            except yaml.YAMLError as exc:
                self.log(
                    f"YAML syntax error. File: {fname} fname. Exc: {exc}",
                    type='fatal'
                )
                rospy.signal_shutdown()
                return

    def callback(self, msg):
        if self.cnt > 0:
            return

        # Read image from msg
        img = np.frombuffer(msg.data, np.uint8) 
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        self.h, self.w = img.shape[:2]
        cv2.imwrite("/data/orig.png", img)
        self.ground2pixel()

        # Undistort camera view
        img = self.process_image(img)

        # Plot on the images
        img = self.render_segments(img)

        if self.cnt == 0:
            cv2.imwrite("/data/save.png", img)

        self.cnt += 1
        # Publish
        # img = self._bridge.cv2_to_compressed_imgmsg(img)
        # self.pub_cam.publish(img)

    def process_image(self, img):
        undist_img = cv2.undistort(img, self.camera_mat, self.distort_coef, None, self.camera_mat)
        cv2.imwrite("/data/undist_img.png", undist_img)
        return undist_img

    def draw_segment(self, image, pt_x, pt_y, color):
        defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0 , 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        image = cv2.line(image, (pt_x[0], pt_y[0]), (pt_x[1], pt_y[1]), (b * 255, g * 255, r * 255), 5)
        return image

if __name__ == "__main__":
    node = Augmenter(node_name="augmented_reality")
    rospy.spin()