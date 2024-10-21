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
        self.state_list = ['landed', 'hovering', 'moving', 'emergency']
        self.drone_state = self.state_list[0]  # Default state is landed
        self.vel_cmd = Twist()

        # Initialize ROS publishers
        self.pubs = {}
        self.init_publishers()

        rospy.loginfo("DroneControl initialized.")

    """ Section 1: Initialization the control topics """

    def init_publishers(self, flip: bool = False) -> None:
        """Initializes the publishers for the given topics."""
        self.pubs['takeoff'] = rospy.Publisher('/bebop/takeoff', Empty,
                                               queue_size=10)
        self.pubs['land'] = rospy.Publisher('/bebop/land', Empty,
                                            queue_size=10)
        self.pubs['reset'] = rospy.Publisher('/bebop/reset', Empty,
                                             queue_size=10)
        self.pubs['cmd_vel'] = rospy.Publisher('/bebop/cmd_vel', Twist,
                                               queue_size=10)
        self.pubs['flattrim'] = rospy.Publisher('/bebop/flattrim', Empty,
                                                queue_size=10)
        self.pubs['flip'] = rospy.Publisher('/bebop/flip', Empty,
                                            queue_size=10)
        rospy.loginfo("Initialized publishers for drone control.")

    """ Section 2: Drone control operations """

    def takeoff(self) -> None:
        """
        Sends a takeoff command to the drone.
        """
        if self.drone_state == self.state_list[3]:
            rospy.logwarn("Drone is in emergency mode.")
        elif self.drone_state == self.state_list[0]:  # Drone is landed
            self.pubs['takeoff'].publish(Empty())
            self.drone_state = self.state_list[1]  # Drone is hovering
            rospy.loginfo("Drone hovering.")
        else:
            rospy.loginfo("Drone already in the air.")

    def land(self) -> None:
        """
        Sends a landing command to the drone.
        """
        if self.drone_state == self.state_list[3]:
            rospy.logwarn("Drone is in emergency mode.")
        elif self.drone_state == self.state_list[2]:  # Drone is moving
            rospy.logwarn("Drone is moving. Stop the drone first.")
        elif self.drone_state == self.state_list[1]:  # Drone is hovering
            self.pubs['land'].publish(Empty())
            self.drone_state = self.state_list[0]  # Drone is landed
            rospy.loginfo("Drone landed.")
        else:
            rospy.loginfo("Drone already landed.")

    def reset(self) -> None:
        """
        Reset the drone's state, typically used after an emergency or crash.
        """
        self.pubs['reset'].publish(Empty())
        self.drone_state = self.state_list[0]  # Drone is landed
        rospy.loginfo("Drone reset.")

    def move(self, linear_x: float, linear_y: float, linear_z: float,
             angular_z: float) -> None:
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
        rospy.loginfo(f"Drone moving with velocities (linear_x={linear_x}, "
                      f"linear_y={linear_y}, linear_z={linear_z}, angular_z={angular_z})")
