import gym
import rospy
import roslaunch
import time
import numpy as np
import os

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose, TransformStamped, PoseStamped
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2, Image
from gym.utils import seeding

import sys
import moveit_commander
import moveit_msgs.msg
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

import dynamic_reconfigure.client

class GazeboSmartBotPincherKinectEnvREAL_ARM(gazebo_env.GazeboEnv):


    def __init__(self):
        print("__init__ of GazeboSmartBotPincherKinectEnvREAL_ARM")
        # Launch the moveit config for the real pincher arm
        # launch file is in gym-gazebo/gym_gazebo/env/assets/launch/
        gazebo_env.GazeboEnv.__init__(self, "GazeboSmartBotPincherKinectREAL_ARM_v0.launch")

        self.action_space = spaces.Discrete(3)  # F,L,R
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

        # Getting Basic Information
        # ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot
        print "============ Reference frame group_arm: %s" % self.group_arm.get_planning_frame()
        print "============ Reference frame group_gripper: %s" % self.group_gripper.get_planning_frame()

        # We can also print the name of the end-effector link for this group
        print "============ End effector group_arm: %s" % self.group_arm.get_end_effector_link()

        # We can get a list of all the groups in the robot
        print "============ Robot Groups:"
        print self.robot.get_group_names()

        # Sometimes for debugging it is useful to print the entire state of the robot.
        print "============ Printing robot state"
        print self.robot.get_current_state()
        print "============"

        # success = self.moveArmToHomePosition()
        # self.printMovingArmSuccess(success)

        self.bridge = CvBridge()

    # __init__

    def discretize_observation(self, data, new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if data.ranges[i] == float('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges, done
    # discretize_observation

    def read_kinect_point_cloud(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/camera/depth/points', PointCloud2, timeout=5)
            except:
                pass
        return data
    # read_kinect_point_cloud

    def read_kinect_depth_image(self, numOfPictures=6, saveImage=False, folder="/home/joel/Pictures/", saveToFile=False):
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # seed


    def step(self, action):

        success = self.moveArmToHomePosition()
        self.printMovingArmSuccess(success)

        # Move the robot arm
        pose_arm = self.group_arm.get_current_pose().pose
        x = pose_arm.position.x + 0.01
        y = pose_arm.position.y + 0.05
        z = pose_arm.position.z - 0.05

        quaternion = pose_arm.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
        print("arm gripper orientation in euler format: roll=%.5f, pitch=%.5f, yaw=%.5f" % (roll, pitch, yaw))  # roll=-0.00487, pitch=-1.56998, yaw=0.00502
        
        x = 0.11
        y = 0.0
        z = 0.028
        roll = 0
        pitch = -3*np.pi/2
        yaw = 0

        # TODO: let the action decide where to move the arm
        # success = self.moveArmToPosition(x, y, z, roll, pitch, yaw)
        # self.printMovingArmSuccess(success)

        # self.moveGripper("grip_closed")

        # # get the reward via text input
        # input_prompt = 'Enter the reward (high if the gripper could grasp the object, low if not.\n-> '
        # reward = int(input(input_prompt))
        # print("received reward = %i" % reward)

        # # rospy.sleep(600)

        # success = self.moveArmToHomePosition()
        # self.printMovingArmSuccess(success)

        # self.moveGripper("grip_open")

        # read kinect point cloud data

        # data = self.read_kinect_point_cloud()
        # f = open("/home/joel/Documents/pointcloud_data_real.txt", 'w')
        # print >>f, data
        numOfPictures = 6
        images = np.empty([480,640,numOfPictures])
        for i in range(0,numOfPictures):
            print("getting image")
            images[:,:,i] = self.read_kinect_depth_image(numOfPictures = 6)
            filename = "/home/joel/Documents/depth_image_real_sliced_" + str(i) + ".jpg"
            cv2.imwrite(filename, images[:,:,i])
            print("depth image written to file")
        image = np.mean(images, axis=2)
        # cutting image so that it only contains region of interest (original size: 480 x 640)
        image = image[240-200:240+20,320-150:320+150]
        # print(image.shape) # (220, 300)
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        # saving depth image
        # np.set_printoptions(threshold='nan')
        # f = open("/home/joel/Documents/depth_image_real.txt", 'w')
        # print >>f, image
        # f.close()
        filename = "/home/joel/Documents/depth_image_real_sliced_" + str("average") + ".jpg"
        cv2.imwrite(filename, image)
        print("depth image written to file")

        rospy.sleep(600)

        # TODO: which state to return (and in which format etc.)
        # state,done = self.discretize_observation(data,5)
        state, done = [1, 2, 3, 4], False

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}
    # step

    def reset(self):
        print("dynamic configure of the kinect camera")

        # dynamic_reconfigure (see http://wiki.ros.org/hokuyo_node/Tutorials/UsingDynparamToChangeHokuyoLaserParameters#PythonAPI
        # and https://answers.ros.org/question/55689/lower-kinect-resolution-for-rgbcamera-using-openni_launch/?answer=55690#post-id-55690)
        kinect_reconfigure_client = dynamic_reconfigure.client.Client("/camera/driver")
        # Depth output mode Possible values are: SXGA (1): 1280x1024, VGA (2): 640x480. SXGA is not supported for depth mode (see http://wiki.ros.org/freenect_camera)
        params = { 'depth_mode' : 2 }
        config = kinect_reconfigure_client.update_configuration(params)

        # read point cloud data
        pointCloud = self.read_kinect_point_cloud()
        print("receved point cloud data in gym-gazebo environment! data.is_dense=", pointCloud.is_dense)

        #state = self.discretize_observation(data,5)
        state = [1, 2, 3, 4, 5, False]

        return state
    # reset

    def close(self):
        print("closing GazeboSmartBotPincherKinectEnvREAL_ARM")
        # When finished, shut down moveit_commander.
        moveit_commander.roscpp_shutdown()
        super(GazeboSmartBotPincherKinectEnvREAL_ARM, self).close()
    # close

    def moveArmToHomePosition(self):
        print("============ going to home position")
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

    def moveArmToPosition(self, x, y, z, roll, pitch, yaw, execute=True):
        print("moving arm to position ", x, y, z, roll, pitch, yaw)
        targetPose = Pose()
        targetPose.position.x = x
        targetPose.position.y = y
        targetPose.position.z = z
        # when working with the orientation, the quaternion has to be normalized, so this doesn't work:
        # targetPose.orientation.x = 0.5
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        targetPose.orientation.x = quaternion[0]
        targetPose.orientation.y = quaternion[1]
        targetPose.orientation.z = quaternion[2]
        targetPose.orientation.w = quaternion[3]
        # self.group_arm.allow_replanning(true)
        self.group_arm.set_pose_target(targetPose)
        planArm = self.group_arm.plan()
        if(execute):
            # success = self.group_arm.execute(planArm)
            success = self.group_arm.go(wait=True)
            return success
    # moveArmToPosition

    def printMovingArmSuccess(self, success):
        if(not success):
            print("======================== FAILED TO MOVE ARM ========================")
        # else:
        #     print("======================== MOVED ARM SUCESSFULLY ========================")
    # printMovingArmSuccess

# class GazeboSmartBotPincherKinectEnvREAL_ARM
