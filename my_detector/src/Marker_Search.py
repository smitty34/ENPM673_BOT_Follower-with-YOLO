from matplotlib.pyplot import table
import rospy
from geometry_msgs.msg import Twist, PoseArray
import darknet_ros_msgs
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox, ObjectCount
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import numpy as np
import message_filters
import sys
import os
from apriltag_ros.msg import AprilTagDetectionArray
from tf.transformations import euler_from_quaternion
import math



robot_x = 0
robot_y = 0
person_x = 0
person_y = 0


class AprilTAG_MODULE:
    Tagdata = {"DX":0.0, "DY":0.0, "DZ":0.0, "AX":0.0, "AY":0.0, "AZ":0.0, "AW":0.0}
    
    def __init__(self):
        self.Tagdata = {"DX":0.0, "DY":0.0, "DZ":0.0, "AX":0.0, "AY":0.0, "AZ":0.0, "AW":0.0}
        self.id = 0
        self.roll, self.pitch, self.yaw = 0,0,0
        
    def set_TagData(self, data):
        for i in data.markers:
            self.id = i.id
            self.Tagdata["DX"] = i.pose.pose.position.x
            self.Tagdata["DY"] = i.pose.pose.position.y
            self.Tagdata["DZ"] = i.pose.pose.position.z

            self.Tagdata["AX"] = i.pose.pose.orientation.x
            self.Tagdata["AY"] = i.pose.pose.orientation.y
            self.Tagdata["AZ"] = i.pose.pose.orientation.z
            self.Tagdata["AW"] = i.pose.pose.orientation.w



            self.roll, self.pitch, self.yaw = euler_from_quaternion([self.Tagdata["AX"], self.Tagdata["AY"], self.Tagdata["AZ"], self.TagData["AW"]])


    def get_yaw(self):
        return self.yaw

    def get_pitch(self):
        return self.pitch

    def get_arctan(self):
        return math.degrees(np.arctan(self.Tagdata["DX"]/ self.Tagdata["DZ"]))

    def get_id(self):
        return self.id

    def get_TagData(self):
        return self.Tagdata

    def get_distance(self):
        return math.sqrt(pow(self.Tagdata["DX"], 2) + pow(self.Tagdata["DX"], 2))

    def Tag_Found(self):
        if self.get_id() == 0 and 0 < self.get_distance() < 0.65:
            return True

        return False

    def finish_T_parking(self):
        if self.get_distance() >= 0.76 and self.get_distance() <= 0.85 and abs(self.get_arctan()) < 10:
            return True

        return False






class Following:
    #480 * 640
    
    def __init__(self, object_names):
        self.center_x = 320
        self.center_y = 240
        self.turn_value = 0.25
        self.linear_value = 0.1
        self.linear_tol = 0.05
        self.linear_coefficient = 0.2
        self.angular_tol = 15
        self.angular_coefficient = 0.0015
        self.object_names = object_names
        self.rate = rospy.Rate(50)
        self.command = Twist()
        #set up publisher and subscriber
        self.xmin = 0
        self.xmax = 640
        self.ymin = 0
        self.ymax = 480
        self.sampling_depth = None
        self.control_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size = 0)
        self.bridge = CvBridge()
        self.depth_msg = None

        self.Apriltag = AprilTAG_MODULE()

        



        # person following filters
        self.person_minimum_area = 13000

        # control buffers
        self.linear_velocity_buffer = 0
        self.linear_velocity_max = 0.3

        '''TODO: define callback function'''
        self.bounding_boxes_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bounding_boxes_callback)
        self.depth_callback = rospy.Subscriber('/camera/depth_registered/image', Image, self.depth_callback)
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.april_tag_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry)
        self.person_sub = rospy.Subscriber('/to_pose_array/leg_detector', PoseArray, self.call_back_leg_detector)


        # self.detection_image_sub = rospy.Subscriber('/darknet_ros/detection_image', BoundingBoxes, self.bounding_boxes_callback)
        # self.depth_sub = rospy.Subscriber('/camera/depth_registered/image', Image, self.depth_callback)
        print('synchronized')

        

    # def depth_callback(self, image):
    #     self.image = image

    def bounding_boxes_callback(self, boundbox_msg):
        print('doing angular callback')
        # depth_msg = rospy.wait_for_message('/camera/depth_registered/image', Image)

        detected = False
        # if we are tracking person, we want the person with the biggest frame area
        if 'person' in self.object_names:
            best_area = float('-inf')
            for i in range(len(boundbox_msg.bounding_boxes)):
                oj = boundbox_msg.bounding_boxes[i]
                area = (oj.ymax - oj.ymin) * (oj.xmax - oj.xmin)
                if (oj.Class in self.object_names) and ((area > best_area) and area > self.person_minimum_area):
                    cur_frame = oj
                    detected = True
                    best_area = area
        else:
            for i in range(len(boundbox_msg.bounding_boxes)):
                if boundbox_msg.bounding_boxes[i].Class in self.object_names:
                    cur_frame = boundbox_msg.bounding_boxes[i]
                    detected = True
                    break   
        if not detected:
            print('not detected')
            return
        #detected object in frame
        else:
            self.xmin = cur_frame.xmin
            self.ymin = cur_frame.ymin
            self.xmax = cur_frame.xmax
            self.ymax = cur_frame.ymax
            name = cur_frame.Class
            self.command = Twist()
            adjusted_angular = self.adjust_orientation(self.xmin, self.ymin, self.xmax, self.ymax)
            if self.sampling_depth != None:
                self.adjust_depth(self.sampling_depth) 
                print(self.command.linear.x)
        if self.command.linear.x != 0:
            self.command.linear.x = min((self.command.linear.x + self.linear_velocity_buffer)/2, self.linear_velocity_max)
        
        self.control_pub.publish(self.command)
        print('linear: ', self.command.linear.x)
        print('angular', self.command.angular.z)
        self.linear_velocity_buffer = self.command.linear.x
        

    # def call_back_leg_detector(self, msg):
    #     global person_x
    #     global person_y

    #     if len(msg.poses) !=0:
    #         person_x = msg.poses[0].position.x
    #         person_y = msg.poses[0].position.y



    # def leg_detect(self):
    #     self.person_sub

    #     angle = atan2((person_y))


    def depth_callback(self, depth_msg):
        self.depth_msg = depth_msg
        cv_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        depth_matrix = np.array(cv_image)
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax

        center = [xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2]
        chord = 5
        new_xmin = center[0] - chord
        new_xmax = center[0] + chord
        new_ymin = center[1] - chord
        new_ymax = center[1] + chord

        sampling_matrix = depth_matrix[new_ymin:new_ymax, new_xmin:new_xmax]
        sampling_depth = np.nanmean(sampling_matrix)

        self.sampling_depth = sampling_depth
        


    def adjust_orientation(self, xmin, ymin, xmax, ymax):
        #adjust the angular twist of the turtlebot to let object be in the center of the image
        xmid_point = xmin + (xmax - xmin)/2
        print('xmid',xmid_point)
        error = xmid_point - self.center_x
        if xmid_point > self.center_x and abs(xmid_point - self.center_x) > self.angular_tol:
            self.command.angular.z = -self.angular_coefficient * error
            return True
        elif xmid_point < self.center_x and abs(xmid_point - self.center_x) > self.angular_tol:
            self.command.angular.z = -self.angular_coefficient * error
            return True
        self.command.angular.z = 0
        
        return False



    def adjust_depth(self, sampling_depth):
        #adjust the depth to let image be of a fixed size
        target_depth = 0.3
        tol = self.linear_tol
        if sampling_depth > target_depth and abs(sampling_depth - target_depth) > tol:
            self.command.linear.x = self.linear_coefficient * (sampling_depth - target_depth)
            return True
        self.command.linear.x = 0
        return False



    def april_tag_callback(self):
        
        self.Apriltag.set_TagData


    
    def not_detected_algo(self):
        #if not detected anything, then just circle
        self.command.angular.z = 0.3
        self.command.linear.x = 0
        rospy.sleep(1)
    
if __name__ == "__main__":

    find_tag = True

    AprilTg = AprilTAG_MODULE

    if AprilTg.Tag_Found():
        print(AprilTg.get_distance())

    

   # follow_object = [sys.argv[1]]
    rospy.init_node('following_node', anonymous=True)
    print('INITIALIZED NODE')
    try:
        following_turtlebot = Following(follow_object)

    except rospy.ROSInterruptException:
        print("exception")
        pass
    following_turtlebot.rate.sleep()
    rospy.spin()