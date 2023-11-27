#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
import numpy as np

depth_image = None
cv_image_norm = None
distance_at_coord = None
distance = None

def depth_image_callback(depth_image_msg):
    global depth_image
    global cv_image_norm
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")
    cv_image_array = np.array(depth_image, dtype = np.dtype('f8'))
    cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)

def coord_callback(data):
    global coord_x
    global coord_y
    global distance_at_coord
    global distance
    coord_x = data.linear.x
    coord_y = data.linear.y
    if depth_image is not None and 0 <= coord_x < depth_image.shape[1] and 0 <= coord_y < depth_image.shape[0]:
        distance = depth_image[round(coord_y), round(coord_x)] / 1000
        distance = "{:.3f}".format(distance)
        pub = rospy.Publisher('distance', String, queue_size=10)
        pub.publish(distance)
        rospy.loginfo(distance)
    else:
        distance_at_coord = None

def subscribe_coords():
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_image_callback)
    rospy.Subscriber('coords', Twist, coord_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('depth_image_processor')
        subscribe_coords()
    except rospy.ROSInterruptException:
        pass


