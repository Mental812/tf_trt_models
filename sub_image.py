#!/usr/bin/env python

from threading import Thread

import cv2

import rospy
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class CImageSubscriber:
  def __init__(self, topic):
    rospy.init_node('test_image_sub', anonymous=True)
    rate = rospy.Rate(10)
    
    self.sub_topic = topic ##rospy.Subscriber(topic, Image, __image_callback)
    self.image = None
    self.sub_thread = Thread(target=self.thread_loop)
    self.sub_thread.start()

  def thread_loop(self):
    while(1):
      self.image_callback()

  def image_callback(self):
    '''Image callback'''
    msg = rospy.wait_for_message(self.sub_topic, Image, None)
    self.image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

  def getImage(self):
    return self.image

if __name__ == "__main__":
  rospy.init_node('test_image_sub', anonymous=True)
  rate = rospy.Rate(10)
  image_sub = CImageSubscriber(topic="/camera/color/image_raw")
  image_pub = rospy.Publisher("/test/image_raw", Image, queue_size=10)

  while not rospy.is_shutdown():
    image = image_sub.getImage()
    cv
    if image is not None:
      image_pub.publish(CvBridge().cv2_to_imgmsg(image, "bgr8"))