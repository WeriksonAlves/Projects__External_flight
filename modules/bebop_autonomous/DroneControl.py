from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
# from typing import List
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
        self.state_list = ['landed', 'takingoff', 'hovering', 'moving',
                           'landing', 'emergency_landing', 'emergency']
        self.drone_state = self.state_list[0]  # Default state is landed
        self.vel_cmd = Twist()
        self.pubs = {}
        self.init_publishers()
        rospy.loginfo("DroneControl initialized.")

    """ Section 1: Initialization the control topics """

    def init_publishers(self, flip: bool = False) -> None:
        """
        Initializes the publishers for the given topics.
        """
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
        if flip:
            self.pubs['flip'] = rospy.Publisher('/bebop/flip', Empty,
                                                queue_size=10)
        rospy.loginfo("Initialized publishers for drone control.")

    """ Section 2: Drone control operations """

    def takeoff(self, simulation: bool = False) -> None:
        """ Sends a takeoff command to the drone."""
        if self.drone_state == self.state_list[0]:  # Drone is landed
            self.pubs['takeoff'].publish(Empty())
            self.drone_state = self.state_list[1]  # Drone is takingoff
            rospy.loginfo("Drone taking off.")
            self.drone_state = self.state_list[2]  # Drone is hovering
            rospy.loginfo("Drone hovering.")
        else:
            rospy.loginfo("Drone already in the air.")

    def land(self, simulation: bool = False) -> None:
        """ Sends a landing command to the drone."""
        if self.drone_state == self.state_list[2]:  # Drone is hovering
            self.pubs['land'].publish(Empty())
            self.drone_state = self.state_list[4]  # Drone is landing
            rospy.loginfo("Drone landing.")
            self.drone_state = self.state_list[0]  # Drone is landed
            rospy.loginfo("Drone landed.")
        else:
            rospy.loginfo("Drone already landed.")

    def reset(self) -> None:
        """
        Reset the drone's state, typically used after an emergency or crash.
        """
        self.pubs['reset'].publish(Empty())
        rospy.loginfo("Drone reset.")
