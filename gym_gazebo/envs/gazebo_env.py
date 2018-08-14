import gym
import rospy
#import roslaunch
import os
import signal
import subprocess
import time
from os import path
from std_srvs.srv import Empty
import random

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile):

        random_number = random.randint(10000, 15000)
        self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_gazebo = str(random_number+1) #os.environ["ROS_PORT_SIM"]
        # os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
        # os.environ["GAZEBO_MASTER_URI"] = "http://localhost:"+self.port_gazebo
        #
        # self.ros_master_uri = os.environ["ROS_MASTER_URI"];

        with open("log.txt", "a") as myfile:
            myfile.write("export ROS_MASTER_URI=http://localhost:"+self.port + "\n")
            myfile.write("export GAZEBO_MASTER_URI=http://localhost:"+self.port_gazebo + "\n")

        #start roscore
        subprocess.Popen(["roscore", "-p", self.port])
        time.sleep(1)
        print ("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets","launch", launchfile)
        if not path.exists(fullpath):
            raise IOError("File "+fullpath+" does not exist")

        subprocess.Popen(["roslaunch","-p", self.port, fullpath])
        print ("Gazebo launched!")

        self.gzclient_pid = 0

    def set_ros_master_uri(self):
        os.environ["ROS_MASTER_URI"] = self.ros_master_uri

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0
    
    
    closeCount = 0

    def close(self):

        # Kill gzclient, gzserver and roscore
        print("gazbebo_env.py: closing the gazebo env, closeCount = " + str(self.closeCount))
        self.closeCount += 1
        
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
            os.wait()

    def configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
