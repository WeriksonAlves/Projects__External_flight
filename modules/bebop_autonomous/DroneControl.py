from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32
import rospy


class DroneControl:
    """
    DroneControl handles basic control operations of the Bebop drone, including
    takeoff, landing, movement, and reset operations via ROS topics.
    """

    def __init__(self):
        """
        Initialize the DroneControl class with publishers and subscribers
        to handle drone operations.
        """
        self.drone_type = 'bebop2'
        self.drone_state = "landed"  # Track drone state (landed, flying, etc.)
        self.vel_cmd = Twist()

        # Initialize ROS publishers
        self.pubs = {}
        self.init_publishers()

        rospy.loginfo("DroneControl initialized.")

    def init_publishers(self) -> None:
        """
        Initialize the necessary ROS publishers for drone control.
        """
        self.pubs['takeoff'] = rospy.Publisher('/bebop/takeoff', Empty, queue_size=10)
        self.pubs['land'] = rospy.Publisher('/bebop/land', Empty, queue_size=10)
        self.pubs['reset'] = rospy.Publisher('/bebop/reset', Empty, queue_size=10)
        self.pubs['cmd_vel'] = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=10)
        self.pubs['flattrim'] = rospy.Publisher('/bebop/flattrim', Empty, queue_size=10)
        # self.pubs['flip'] = rospy.Publisher('/bebop/flip', Empty, queue_size=10)
        rospy.loginfo("Initialized publishers for drone control.")

    def takeoff(self) -> None:
        """
        Command the drone to take off.
        """
        if self.drone_state == "landed":
            self.pubs['takeoff'].publish(Empty())
            self.drone_state = "flying"
            rospy.loginfo("Drone taking off.")

    def land(self) -> None:
        """
        Command the drone to land.
        """
        if self.drone_state == "flying":
            self.pubs['land'].publish(Empty())
            self.drone_state = "landed"
            rospy.loginfo("Drone landing.")

    def reset(self) -> None:
        """
        Reset the drone's state, typically used after an emergency or crash.
        """
        self.pubs['reset'].publish(Empty())
        rospy.loginfo("Drone reset.")

    def move(self, linear_x: float, linear_y: float, linear_z: float, angular_z: float) -> None:
        """
        Command the drone to move based on velocity inputs.

        :param linear_x: Forward/backward velocity.
        :param linear_y: Left/right velocity.
        :param linear_z: Up/down velocity.
        :param angular_z: Rotational velocity around the Z-axis (yaw).
        """
        self.vel_cmd.linear.x = linear_x
        self.vel_cmd.linear.y = linear_y
        self.vel_cmd.linear.z = linear_z
        self.vel_cmd.angular.z = angular_z
        self.pubs['cmd_vel'].publish(self.vel_cmd)
        rospy.loginfo(f"Drone moving with velocities (linear_x={linear_x}, linear_y={linear_y}, linear_z={linear_z}, angular_z={angular_z})")

    def flattrim(self) -> None:
        """
        Command the drone to perform a flat trim calibration.
        """
        self.pubs['flattrim'].publish(Empty())
        rospy.loginfo("Flat trim calibration command issued.")

    def flip(self, direction: str) -> None:
        """
        Command the drone to flip in a specified direction.

        :param direction: Direction of the flip (e.g., forward, backward, left, right).
        """
        if direction.lower() in ['forward', 'backward', 'left', 'right']:
            self.pubs['flip'].publish(Empty())  # Placeholder for actual implementation
            rospy.loginfo(f"Drone flip command issued in {direction} direction.")
        else:
            rospy.logwarn(f"Invalid flip direction: {direction}. Use 'forward', 'backward', 'left', or 'right'.")

    def navigate_home(self) -> None:
        """
        Command the drone to navigate back to its home position.
        """
        rospy.loginfo("Drone navigating home (command not implemented yet).")
        # Implement navigation to home position logic here.

    def emergency_stop(self) -> None:
        """
        Trigger an emergency stop for the drone.
        """
        rospy.logwarn("Emergency stop triggered.")
        self.reset()

    def pause(self) -> None:
        """
        Pause any ongoing autoflight operation.
        """
        rospy.loginfo("Autoflight paused (command not implemented yet).")

    def stop(self) -> None:
        """
        Stop the drone's current movement and hover.
        """
        self.move(0.0, 0.0, 0.0, 0.0)
        rospy.loginfo("Drone stopped and is hovering.")

    def set_exposure(self, exposure_value: float) -> None:
        """
        Set the drone camera's exposure value.

        :param exposure_value: A float value to adjust exposure (-3.0 to 3.0).
        """
        try:
            exposure_msg = Float32(data=exposure_value)
            self.pubs['set_exposure'].publish(exposure_msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to set exposure: {e}")

    def start_control(self) -> None:
        """
        Start the control loop and keep the ROS node active.
        """
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node('bebop_drone_control', anonymous=True)
    drone_control = DroneControl()

    # Example commands
    drone_control.takeoff()
    rospy.sleep(2)  # Wait for 2 seconds after takeoff
    drone_control.move(0.5, 0.0, 0.0, 0.0)  # Move forward
    rospy.sleep(2)
    drone_control.stop()  # Stop the drone and hover
    rospy.sleep(2)
    drone_control.land()
