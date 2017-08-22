#!/usr/bin/env python
# This file is responsible for bridging ROS to the ObjectDetector class (built with PyCaffe)

from __future__ import division

import sys

import cv2
import rospy
from lib.utils.timer import Timer
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rail_faster_rcnn.msg import Object, Detections

import object_detector

# Debug Helpers
FAIL_COLOR = '\033[91m'
ENDC_COLOR = '\033[0m'


def eprint(error):
    sys.stderr.write(
        FAIL_COLOR
        + type(error).__name__
        + ": "
        + error.message
        + ENDC_COLOR
    )
# End Debug Helpers


class RCNNDetector(object):
    """
    This class takes in image data and finds / annotates objects within the image
    """

    def __init__(self):
        rospy.init_node('rcnn_detector_node')
        self.objects1 = []
        self.objects2 = []
        self.objects = []
        self.keypoint_arrays = []
        self.image_datastream = None
        self.input_image = None
        self.bridge = CvBridge()
        self.detector = object_detector.RCNNDetector()
        self.debug = rospy.get_param('~debug', default=False)
        self.image_sub_topic_name = rospy.get_param('~image_sub_topic_name', default='/kinect/hd/image_color_rect')

    def _draw_bb(self, image, bounding_box, color):
        start_x = bounding_box['x']
        start_y = bounding_box['y']
        end_x = start_x + bounding_box['w']
        end_y = start_y + bounding_box['h']
        cv2.rectangle(image,
                      (start_x, start_y),
                      (end_x, end_y),
                      color=color,
                      thickness=3)
        return image

    def _parse_image(self, image_msg):
        """
        Take in an image and draw a bounding box within it
        Publishes bounding box data onwards
        :param image_msg: Image data
        :return: None
        """

        header = image_msg.header

        try:
            image_cv = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print e
            return
        down_shift = int(len(image_cv)*.527)
        bottom_img = down_shift+230
        table_shift = int(len(image_cv[0])*.455)
        table_end = table_shift+726
        # Test on the right table hd crop
        img2 = image_cv[down_shift:bottom_img, table_shift:table_end, :]

        # Detect all object classes and regress object bounds
        right_objects = []
        objects_back = self.detector.find_objects(img2)
        for obj_list in objects_back:
            for obj in obj_list:
                adjusted_x1 = obj[0] + table_shift
                adjusted_y1 = obj[1] + down_shift
                width = obj[2] - obj[0]
                height = obj[3] - obj[1]
                cls = obj[4]
                right_objects.append([adjusted_x1, adjusted_y1, width, height, cls])

        # timer.toc()
        # print ('Detection took {:.3f}s for '
        #        '{:d} object proposals').format(timer.total_time, len(self.objects2))
        self.objects = right_objects

        if self.debug:
            for obj in self.objects:
                x1 = int(obj[0])
                y1 = int(obj[1])
                width = int(obj[2])
                height = int(obj[3])
                image_cv = self._draw_bb(image_cv, {'x': x1,
                                                    'y': y1,
                                                    'w': width,
                                                    'h': height}, (0, 255, 0))
            try:
                image_msg = self.bridge.cv2_to_imgmsg(image_cv, "bgr8")
            except CvBridgeError as e:
                print e

            image_msg.header = header
            self.image_pub.publish(image_msg)
        #### END DEBUG ####

        # Instantiate detections object
        obj_arr = Detections()
        obj_arr.header = header
        # obj_arr.objects = self.objects
        # For each object / keypoint set found in the image:

        for bbox_obj in self.objects:
            if len(bbox_obj) < 1:
                continue
            msg = Object()
            msg.object_id = bbox_obj[4]
            msg.top_left_x = int(bbox_obj[0])
            msg.top_left_y = int(bbox_obj[1])
            msg.bot_right_x = int(bbox_obj[2]) + int(bbox_obj[0])
            msg.bot_right_y = int(bbox_obj[3]) + int(bbox_obj[1])
            obj_arr.objects.append(msg)

        self.object_pub.publish(obj_arr)

    def run(self,
            pub_image_topic='/object_detector/debug/obj_image',
            pub_object_topic='/object_detector/objects'):
        rospy.Subscriber(self.image_sub_topic_name, Image, self._parse_image) # subscribe to sub_image_topic and callback parse
        if self.debug:
            self.image_pub = rospy.Publisher(pub_image_topic, Image, queue_size=2) # image publisher
        self.object_pub = rospy.Publisher(pub_object_topic, Detections, queue_size=2) # objects publisher
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = RCNNDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
