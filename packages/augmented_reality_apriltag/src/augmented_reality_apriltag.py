#!/usr/bin/env python3
import yaml

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.srv import ChangePattern

import numpy as np
import cv2
from cv_bridge import CvBridge
from dt_apriltags import Detector


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


class AprilTagAR(DTROS):
    def __init__(self, node_name="augmented_reality_apriltag"):
        super(AprilTagAR, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )

        # Get arguments from a launch file
        self._veh = rospy.get_param("~veh")
        self._int_path = rospy.get_param("~int_file")
        self._ext_path = rospy.get_param("~ext_file")

        self.counts = 0

        # Prepare CV bridge
        self.bridge = CvBridge()

        # Prepare apriltag detector
        self.apriltag_detector = Detector(families="tag36h11",
                                          nthreads=1)

        # Initialize matrices
        self.camera_mat, self.distort_coef, self.proj_mat, self.hom_mat = \
            parse_calib_params(self._int_path, self._ext_path)
        
        # Service client
        rospy.wait_for_service(f"/{self._veh}/led_emitter_node/set_pattern")
        self.srv_led = rospy.ServiceProxy(f"/{self._veh}/led_emitter_node/set_pattern", ChangePattern)

        # Subscriber
        self.sub_cam = rospy.Subscriber(
            f"/{self._veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_cam
        )

        # Publisher
        self.pub_view = rospy.Publisher(
            f"/{self._veh}/{node_name}/image/compressed",
            CompressedImage,
            queue_size=10
        )

    def undistort(self, img):
        return cv2.undistort(img,
                             self.camera_mat,
                             self.distort_coef,
                             None,
                             self.camera_mat)

    def draw_detect_results(self, img, tags):
        n_detections = len(tags)
        which_tag = ''
        for tag in tags:
            # Enumerate through the detection results
            (ptA, ptB, ptC, ptD) = tag.corners

            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            # Determine the decoded ID of tag, and decide the color of bbox 
            id = tag.tag_id

            # TODO: determine the associated ID per sign
            tag_name = "Stop sign"
            which_tag = 's'
            col = (0, 0, 255)   # stop sign (default)
            if id in [153, 58, 133, 62]:
                tag_name = "T-intersection"
                col = (255, 0, 0)   # T-intersection sign
                which_tag = 't'
            elif id in [201, 93, 94, 200]:
                tag_name = "UofA"
                col = (0, 255, 0)   # UofA sign
                which_tag = 'a'
            
            # Draw the bbox
            cv2.line(img, ptA, ptB, col, 2)
            cv2.line(img, ptB, ptC, col, 2)
            cv2.line(img, ptC, ptD, col, 2)
            cv2.line(img, ptD, ptA, col, 2)

            # Draw the center dot and name of tag
            cx, cy = int(tag.center[0]), int(tag.center[1])
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, tag_name, (cx, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

        return img, n_detections, which_tag

    def cb_cam(self, msg):
        if self.counts % 10 == 0:
            # Reconstruct byte sequence to image
            img = np.frombuffer(msg.data, np.uint8) 
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            # Undistort an image
            img = self.undistort(img)

            # Convert image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect an apriltag
            tags = self.apriltag_detector.detect(gray_img)
            
            # Plot detected features on image
            img, cnt, which_tag = self.draw_detect_results(img, tags)

            # Change LED colors bases on the detected results
            led_msg = String()
            if cnt == 0:
                # white led
                led_msg.data = "WHITE"
            elif cnt == 1 and which_tag == 's':
                # red led
                led_msg.data = "RED"
            elif cnt == 1 and which_tag == 't':
                # blue led
                led_msg.data = "BLUE"
            elif cnt == 1 and which_tag == 'a':
                # green led
                led_msg.data = "GREEN"
            else:
                # lights off (multiple detections)
                led_msg.data = "LIGHT_OFF"
            self.srv_led(led_msg)

            # Encode processed image into byte sequence
            pub_img = self.bridge.cv2_to_compressed_imgmsg(img)
            self.pub_view.publish(pub_img)
            self.counts = 0

        self.counts += 1

if __name__ == "__main__":
    apriltag_ar = AprilTagAR()
    rospy.spin()