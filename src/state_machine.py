# Python
from __future__ import annotations

# ROS
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class State(object):

    def __init__(self):
        pass

    @property
    def name(self):
        return ''

    def enter(self, machine: StateMachine):
        pass

    def exit(self, machine: StateMachine):
        pass

    def update(self, machine: StateMachine):
        pass

    def set_bumper_state(self, bumper_state: int, bumper_location: int):
        self.bumper_state = bumper_state
        self.bumper_location = bumper_location

    def set_odometry(self, odometry: Odometry):
        self.odometry = odometry

    def set_velocity(self, velocity: Twist):
        self.velocity = velocity

class StateMachine(object):

    def __init__(self):
        self.state = None
        self.states = {}

    def add_state(self, state: State):
        self.states[state.name] = state

    def go_to_state(self, state_name: str):
        if self.state:
            rospy.logdebug('TrajectorySupervisor - StateMachine: Exiting %s', self.state.name)
            self.state.exit(self)
        self.state = self.states[state_name]
        rospy.logdebug('TrajectorySupervisor - StateMachine: Entering %s', self.state.name)
        self.state.enter(self)

    def update(self):
        if self.state:
            rospy.logdebug('TrajectorySupervisor - StateMachine: Updating %s', self.state.name)
            self.state.update(self)
