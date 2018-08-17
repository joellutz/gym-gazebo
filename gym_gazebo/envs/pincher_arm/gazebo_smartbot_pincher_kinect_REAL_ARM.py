import gym
import rospy
import roslaunch
import time
import numpy as np
import os
from os.path import expanduser

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose, TransformStamped, PoseStamped
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2, Image
from gym.utils import seeding

import sys
import moveit_commander
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import tf
from gazebo_msgs.msg import LinkStates, ModelStates, LinkState, ContactsState, ContactState
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnModel
import random
from rosgraph_msgs.msg import Clock
from numpy import matrix
from math import cos, sin
from cv_bridge import CvBridge, CvBridgeError
import cv2
import psutil

import dynamic_reconfigure.client

# NOT WELL TESTED!

class GazeboSmartBotPincherKinectEnvREAL_ARM(gazebo_env.GazeboEnv):
    
    # some algorithms (like the ddpg from /home/joel/Documents/gym-gazebo/examples/pincher_arm/smartbot_pincher_kinect_ddpg.py)
    # currently assume that the observation_space has shape (x,) instead of (220,300), so for those algorithms set this to True
    flattenImage = True

    state = np.array([])
    imageCount = 0
    home = expanduser("~")

    def __init__(self):
        print("__init__ of GazeboSmartBotPincherKinectEnvREAL_ARM")
        # Launch the moveit config for the real pincher arm
        # launch file is in gym-gazebo/gym_gazebo/env/assets/launch/
        gazebo_env.GazeboEnv.__init__(self, "GazeboSmartBotPincherKinectREAL_ARM_v0.launch")

        # the state space (=observation space) are all possible depth images of the kinect camera
        if(self.flattenImage):
            self.observation_space = spaces.Box(low=0, high=255, shape=[220*300], dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=[220,300], dtype=np.uint8)

        # the action space are all possible positions & orientations (6-DOF),
        # which are bounded in the area in front of the robot arm where an object can lie (see reset())
        boundaries_xAxis = [0.04, 0.3]      # box position possiblities: (0.06, 0.22)
        boundaries_yAxis = [-0.25, 0.25]    # box position possiblities: (-0.2, 0.2)
        boundaries_zAxis = [0.015, 0.05]    # box z-position: ca. 0.04
        boundaries_roll = [0, 2*np.pi]
        boundaries_pitch = [0, 2*np.pi]
        boundaries_yaw = [0, 2*np.pi]

        low = np.array([boundaries_xAxis[0], boundaries_yAxis[0], boundaries_zAxis[0], boundaries_roll[0], boundaries_pitch[0], boundaries_yaw[0]])
        high = np.array([boundaries_xAxis[1], boundaries_yAxis[1], boundaries_zAxis[1], boundaries_roll[1], boundaries_pitch[1], boundaries_yaw[1]])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)

        self.seed()

        rospy.sleep(2)

        # MoveIt variables
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node('move_group_python_interface', anonymous=True) # can't init another node, there is already one initialized in the superclass
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_arm = moveit_commander.MoveGroupCommander("arm")
        self.group_gripper = moveit_commander.MoveGroupCommander("gripper")
        # used to be: base_link, also possible: world (if defined in the urdf.xarco file)
        self.REFERENCE_FRAME = 'world'
        # Allow some leeway in position (meters) and orientation (radians)
        self.group_arm.set_goal_position_tolerance(0.01)
        self.group_arm.set_goal_orientation_tolerance(0.01)
        # Allow replanning to increase the odds of a solution
        self.group_arm.allow_replanning(True)
        # Set the right arm reference frame
        self.group_arm.set_pose_reference_frame(self.REFERENCE_FRAME)
        # Allow 5 seconds per planning attempt
        self.group_arm.set_planning_time(5)

        self.bridge = CvBridge()

    # __init__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # seed

    def reset(self):
        print("dynamic configure of the kinect camera")

        # dynamic_reconfigure (see http://wiki.ros.org/hokuyo_node/Tutorials/UsingDynparamToChangeHokuyoLaserParameters#PythonAPI
        # and https://answers.ros.org/question/55689/lower-kinect-resolution-for-rgbcamera-using-openni_launch/?answer=55690#post-id-55690)
        kinect_reconfigure_client = dynamic_reconfigure.client.Client("/camera/driver")
        # Depth output mode Possible values are: SXGA (1): 1280x1024, VGA (2): 640x480. SXGA is not supported for depth mode (see http://wiki.ros.org/freenect_camera)
        params = { 'depth_mode' : 2 }
        config = kinect_reconfigure_client.update_configuration(params)

        # get depth image from kinect camera
        image = self.read_kinect_depth_image(saveImage=True)

        self.state = image
        return self.state
    # reset

    def close(self):
        print("closing GazeboSmartBotPincherKinectEnvREAL_ARM")
        # When finished, shut down moveit_commander.
        moveit_commander.roscpp_shutdown()
        super(GazeboSmartBotPincherKinectEnvREAL_ARM, self).close()
    # close

    def step(self, action):

        success = self.moveArmToHomePosition()
        self.printMovingArmSuccess(success)

        # Move the robot arm
        # x = 0.11
        # y = 0.0
        # z = 0.028
        # roll = 0
        # pitch = np.pi/2
        # yaw = 0
        x = action[0].astype(np.float64)
        y = action[1].astype(np.float64)
        z = action[2].astype(np.float64)
        roll = action[3].astype(np.float64)
        pitch = action[4].astype(np.float64)
        yaw = action[5].astype(np.float64)
        text = "moving arm to position %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % (x, y, z, roll, pitch, yaw)
        print(text)
        pose = self.createPose(x, y, z, roll, pitch, yaw)
        success = self.moveArmToPose(pose)
        self.printMovingArmSuccess(success, printSuccess=True, printFailure=True)

        self.moveGripper("grip_closed")

        # get the reward via text input
        input_prompt = 'Enter the reward (high if the gripper could grasp the object, low if not.\n-> '
        reward = int(input(input_prompt))
        print("received reward = %i" % reward)

        # return arm to home position
        success = self.moveArmToHomePosition()
        self.printMovingArmSuccess(success, printSuccess=True, printFailure=True, target="TO HOME POSITION")

        self.moveGripper("grip_open")

        # get depth image from kinect camera (= next state)
        image = self.read_kinect_depth_image(saveImage=True)

        self.state = image
        done = False
        info = {}

        return self.state, reward, done, info
    # step

    def read_kinect_depth_image(self, numOfPictures=6, saveImage=False, saveToFile=False):
        folder = self.home + "/Pictures/"
        images = np.empty([480,640,numOfPictures])
        for i in range(0,numOfPictures):
            print("getting image nr. {}".format(i))
            data = None
            cv_image = None
            while(data is None or cv_image is None):
                try:
                    data = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5)
                    # encoding of simulated depth image: 32FC1
                    # converting & normalizing image, so that simulated & real depth image have the same encoding & value range
                    # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
                    cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
                except rospy.exceptions.ROSException:
                    # print("failed to read depth image from simulated kinect camera")
                    # print(e)
                    pass
                except CvBridgeError:
                    # print("failed to convert depth image into another encoding")
                    # print(e) 
                    pass
            # while
            
            image = np.array(cv_image, dtype=np.float)
            # print(type(image)) # <type 'numpy.ndarray'>
            # print(image.shape) # (480, 640)
            # cutting image so that it only contains region of interest (original size: 480 x 640)
            images[:,:,i] = image[240-200:240+20,320-150:320+150]
            # print(image.shape) # (220, 300)
            
            if(saveImage):
                self.saveImage(images[:,:,i], folder, i, averagedImage=False, saveToFile=saveToFile)
            # if
        # for

        image = np.mean(images, axis=2)
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)

        # round to integer values
        image = np.round(image).astype(np.uint8)
        
        if(saveImage):
            self.saveImage(image, folder, averagedImage=True, saveToFile=saveToFile)

        if(self.flattenImage):
            return image.flatten()
        else:
            return image
    # read_kinect_depth_image

    def saveImage(self, image, folder, i=0, averagedImage=False, saveToFile=False):
        # saving depth image
        filename = folder + "depth_image_real_" + str(self.imageCount)
        if(averagedImage):
            filename += "_averaged"
        else:
            filename += "." + str(i)
        if(saveToFile):
            np.set_printoptions(threshold="nan")
            f = open(filename + ".txt", "w")
            print >>f, image
            f.close()
        try:
            cv2.imwrite(filename + ".jpg", image)
        except Exception as e:
            print(e)
            return False
        return True
    # saveImage

    def createPose(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        """ Creates a Pose object with the given position & orientation parameters. """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        # when working with the orientation, the quaternion has to be normalized, so this doesn't work:
        # targetPose.orientation.x = 0.5
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        return pose
    # createPose

    def moveArmToHomePosition(self):
        self.group_arm.set_named_target("right_up")
        planHome = self.group_arm.plan()
        success = self.group_arm.go(wait=True)
        return success
        # arm_joint_values = self.group_arm.get_current_joint_values()
        # if(not all(joint_values == 0.0 for joint_values in arm_joint_values)):
        #     print "============ going to home position"
        #     self.group_arm.clear_pose_targets()
        #     for i in range(len(arm_joint_values)):
        #         arm_joint_values[i] = 0.0
        #     self.group_arm.set_joint_value_target(arm_joint_values)
        #     plan0 = self.group_arm.plan()
        #     # success = self.group_arm.execute(plan0)
        #     success = self.group_arm.go(wait=True)
        #     return success
    # moveArmToHomePosition

    def moveGripper(self, namedPose):
        """ grip_open, grip_closed, grip_mid """
        print("============ moving gripper to pose: %s" % namedPose)
        self.group_gripper.set_named_target(namedPose)
        planGripper = self.group_gripper.plan()
        self.group_gripper.execute(planGripper)
        # gripper always returns failed execution, but it works nevertheless, so there's no point in returning the success boolean
    # moveGripper

    def moveArmToPose(self, targetPose, execute=True):
        self.group_arm.set_pose_target(targetPose)
        planArm = self.group_arm.plan()
        if(execute):
            # success = self.group_arm.execute(planArm)
            success = self.group_arm.go(wait=True)
            return success
    # moveArmToPosition

    def printMovingArmSuccess(self, success, printFailure=False, printSuccess=False, target=""):
        if(not success and printFailure):
            print("********************************************* FAILED TO MOVE ARM " + target + " *********************************************")
        elif(success and printSuccess):
            print("********************************************* MOVED ARM SUCCESSFULLY " + target + " *********************************************")
    # printMovingArmSuccess

# class GazeboSmartBotPincherKinectEnvREAL_ARM
