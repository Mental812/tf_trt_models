"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  In
addition, this Camera class is further extended to take a video
file or an image file as input.
"""
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2
import numpy as np
import cv2


class Camera():
    def __init__(self):
        self.is_opened = False
        self.img_width = 640
        self.img_height = 480
        self.cap = None

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'
        self.cap = cv2.VideoCapture(6)
        # ignore image width/height settings here
        #ic = realsense_sub()
        #rospy.init_node('realsense_ObjectDetection_sub', anonymous=True)
       # rospy.spin()
   

    def read(self):
        _, img = self.cap.read()
        if img is None:
            #logging.warning('grab_img(): cap.read() returns None...')
            # looping around
            self.cap.release()
            _, img = cv_realsense_img
        return img

    def release(self):
        if self.cap != 'OK':
            self.cap.release()