"""
DroneCamera
    Purpose: 
        This class handles camera operations, including capturing raw images, managing camera orientation, and controlling exposure settings.

    Topics:
        /bebop/image_raw
        /bebop/image_raw/compressed
        /bebop/camera_control
        /bebop/states/ardrone3/CameraState/Orientation
        /bebop/set_exposure
        /bebop/snapshot

    Responsibilities:
        Capture and process images for gesture recognition.
        Interface with any image compression, depth sensing, or stream encoding.
        Handle camera orientation and exposure settings.
"""

#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

class DroneCamera:
    def __init__(self):
        # Initialize the ROS node for the camera
        rospy.init_node('bebop_drone_camera', anonymous=True)

        # Camera image topics
        self.image_raw_sub = rospy.Subscriber('/bebop/image_raw/compressed', CompressedImage, self.image_callback)
        
        # Camera control topics
        self.camera_control_pub = rospy.Publisher('/bebop/camera_control', Twist, queue_size=10)
        self.snapshot_pub = rospy.Publisher('/bebop/snapshot', Empty, queue_size=10)
        self.set_exposure_pub = rospy.Publisher('/bebop/set_exposure', Empty, queue_size=10)

        # Variable to store image
        self.current_image = None

    def image_callback(self, data):
        """
        Callback function to process the compressed image data.
        """
        try:
            # Process the compressed image (data.data is the actual image bytes)
            self.current_image = data.data
            # You can decode and process this image using OpenCV if needed
            rospy.loginfo("Received an image frame")
        except Exception as e:
            rospy.logerr("Error processing image: {}".format(e))

    def move_camera(self, tilt=0.0, pan=0.0):
        """
        Method to control the camera orientation.
        :param tilt: The tilt value to set for the camera (vertical movement).
        :param pan: The pan value to set for the camera (horizontal movement).
        """
        camera_control_msg = Twist()
        camera_control_msg.angular.y = tilt  # Control the tilt of the camera
        camera_control_msg.angular.z = pan   # Control the pan of the camera
        self.camera_control_pub.publish(camera_control_msg)
        rospy.loginfo(f"Moving camera - Tilt: {tilt}, Pan: {pan}")

    def take_snapshot(self):
        """
        Command the drone to take a snapshot.
        """
        self.snapshot_pub.publish(Empty())
        rospy.loginfo("Snapshot command sent")

    def set_exposure(self, exposure_value):
        """
        Set the camera exposure to a specific value.
        :param exposure_value: The value of the exposure to set.
        """
        # This assumes some exposure command will be published.
        self.set_exposure_pub.publish(Empty())  # Modify as per the actual exposure message type.
        rospy.loginfo(f"Exposure set to {exposure_value}")

    def get_current_image(self):
        """
        Returns the most recent image captured by the drone.
        """
        return self.current_image

    def start_camera_stream(self):
        """
        Starts the camera stream by subscribing to the image topic.
        """
        rospy.spin()  # Keep the camera streaming until stopped

if __name__ == '__main__':
    try:
        camera = DroneCamera()
        rospy.loginfo("Drone camera system initialized")
        camera.start_camera_stream()
    except rospy.ROSInterruptException:
        pass
