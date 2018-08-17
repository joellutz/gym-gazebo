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



class GazeboSmartBotPincherKinectEnv(gazebo_env.GazeboEnv):

    rewardSuccess = 500
    rewardFailure = 0
    rewardUnreachablePosition = -5
    # when set to True, the reward will be rewardSuccess if gripper could grasp the object, rewardFailure otherwise
    # when set to False, the reward will be calculated from the distance between the gripper & the position of success
    binaryReward = False
    
    # some algorithms (like the ddpg from /home/joel/Documents/gym-gazebo/examples/pincher_arm/smartbot_pincher_kinect_ddpg.py)
    # currently assume that the observation_space has shape (x,) instead of (220,300), so for those algorithms set this to True
    flattenImage = True
    
    # how many times reset() has to be called in order to move the ObjectToPickUp to a new (random) position
    randomPositionAtResetFrequency = 50
    resetCount = 0

    currentPoseOfObject = Pose()
    state = np.array([])
    imageCount = 0
    home = expanduser("~")

    def __init__(self):
        """ Initializes the environment and starts the gazebo simulation. """
        # print("__init__ of GazeboSmartBotPincherKinectEnv")
        # Launch the simulation with the given launch file name
        # launch file is in gym-gazebo/gym_gazebo/env/assets/launch/
        gazebo_env.GazeboEnv.__init__(self, "GazeboSmartBotPincherKinect_v0.launch")
        self.link_state_pub = rospy.Publisher("/gazebo/set_link_state", LinkState, queue_size=5)
        self.unpause_proxy = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause_proxy = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

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

        # test with symmetric boundaries
        # boundaries_xAxis = [-0.3, 0.3]
        # boundaries_yAxis = [-0.25, 0.25]
        # boundaries_zAxis = [-0.5, 0.5]
        # boundaries_roll = [-np.pi, np.pi]
        # boundaries_pitch = [-np.pi, np.pi]
        # boundaries_yaw = [-np.pi, np.pi]

        low = np.array([boundaries_xAxis[0], boundaries_yAxis[0], boundaries_zAxis[0], boundaries_roll[0], boundaries_pitch[0], boundaries_yaw[0]])
        high = np.array([boundaries_xAxis[1], boundaries_yAxis[1], boundaries_zAxis[1], boundaries_roll[1], boundaries_pitch[1], boundaries_yaw[1]])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)
        
        self.seed()

        # setting up the scene
        print("inserting table plane")
        self.insertObject(0.4, 0, 0.025/2,
            self.home + "/Documents/gazebo-models/tablePlane/tablePlane.sdf", "tablePlane")

        print("inserting kinect sensor")
        self.insertObject(0.03, 0, 0.5,
            self.home + "/Documents/gazebo-models/kinect_ros/model.sdf", "kinect_ros", roll=0, pitch=np.pi/2, yaw=0)

        print("inserting object to pick up")
        self.insertObject(0.1, 0, 0.1,
            self.home + "/Documents/gazebo-models/objectToPickUp/objectToPickUp.sdf", "objectToPickUp")

        # initializing MoveIt variables
        # it's necessary to unpause the simulation in order to initialize the MoveGroupCommander, printing robot state etc.
        self.unpause_simulation()
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node("move_group_python_interface", anonymous=True) # can't init another node, there is already one initialized in the superclass
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_arm = moveit_commander.MoveGroupCommander("arm")
        # used to be: base_link, also possible: world (if defined in the urdf.xarco file)
        self.REFERENCE_FRAME = "world"
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

        self.pause_simulation()

    # __init__

    def seed(self, seed=None):
        """ Seeds the environment (for replicating the pseudo-random processes). """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # seed

    def reset(self):
        """ Resets the state of the environment and returns an initial observation."""
        self.reset_simulation()

        # Unpause simulation to set position of ObjectToPickUp & make observation
        self.unpause_simulation()
        if(self.resetCount % self.randomPositionAtResetFrequency == 0):
            # print("setting random position of object to pick up")
            x = random.uniform(0.06, 0.22)
            y = random.uniform(-0.2, 0.2)
            z = random.uniform(0.05, 0.1)
            roll = 0
            pitch = 0
            yaw = random.uniform(0, np.pi)
            # self.currentPoseOfObject = self.createPose(x,y,z,roll,pitch,yaw)
            # set defined position for testing purposes
            self.currentPoseOfObject = self.createPose(x=0.1, y=0, z=0.05, roll=0, pitch=0, yaw=0)
            self.setPoseOfObjectToPickUp(self.currentPoseOfObject)
        self.resetCount += 1
        
        # get depth image from kinect camera
        image = self.read_kinect_depth_image(saveImage=True)
        self.pause_simulation()

        self.state = image
        return self.state
    # reset

    def close(self):
        """ Closes the environment and shuts down the simulation. """
        print("closing GazeboSmartBotPincherKinectEnv")
        # When finished, shut down moveit_commander.
        moveit_commander.roscpp_shutdown()
        super(GazeboSmartBotPincherKinectEnv, self).close()
    # close

    def step(self, action):
        """ Executes the action (i.e. moves the arm to the pose) and returns the reward and a new state (depth image). """

        self.unpause_simulation()

        # reset position of object in case it got moved
        self.setPoseOfObjectToPickUp(self.currentPoseOfObject)

        # get the position of the ObjectToPickUp
        pickup_pose_old, gripper_right_position, gripper_left_position = self.getPoseOfObjectAndGripper()

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
        self.printMovingArmSuccess(success, printSuccess=True)

        if(not success):
            # unreachable position has been selected by the RL algorithm
            reward = self.rewardUnreachablePosition
            done = False
            info = {"couldMoveArm" : False}
            # take a new picture (should be very similar like the one before)
            # but don't return previously captured image (= same state as before),
            # as this results in very bad exploration
            self.state = self.read_kinect_depth_image(saveImage=True)
            return self.state, reward, done, info

        pickup_pose, gripper_right_position, gripper_left_position = self.getPoseOfObjectAndGripper()

        # determine if gripper could grasp the ObjectToPickUp
        reward = self.calculateReward(pickup_pose_old, pickup_pose, gripper_right_position, gripper_left_position)

        # return arm to home position
        success = self.moveArmToHomePosition()
        self.printMovingArmSuccess(success, printFailure=True, target="TO HOME POSITION")
        
        # set the ObjectToPickUp where it was before (at the beginning of the step() function)
        self.setPoseOfObjectToPickUp(pickup_pose_old)

        # get depth image from kinect camera (= next state)
        image = self.read_kinect_depth_image(saveImage=True)

        # print("going to sleep")
        # rospy.sleep(600)

        self.pause_simulation()

        self.state = image
        done = False # can't be set to true, as then the IK solver of the arm (in MoveIt) fails to generate a plan
        info = {"couldMoveArm" : True}

        return self.state, reward, done, info
    # step

    def unpause_simulation(self):
        """ Unpauses the gazebo simulation. """
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
            print(e)
    # unpause_simulation

    def pause_simulation(self):
        """ Pauses the gazebo simulation. """
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
            print(e)
    # pause_simulation

    def reset_simulation(self):
        """ Resets the gazebo simulation. """
        try:
            rospy.wait_for_service("/gazebo/reset_simulation", timeout=5)
            self.reset_proxy()
        except (rospy.ServiceException, rospy.ROSException) as e:
            print("/gazebo/reset_simulation service call failed")
            print(e)
            return None
    # reset_simulation

    def read_kinect_depth_image(self, setRandomPixelsToZero=False, saveImage=False, saveToFile=False):
        """ Reads the depth image from the kinect camera, adds Gaussian noise, crops, normalizes and saves it. """
        folder = self.home + "/Pictures/"
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
        image = image[240-170:240+50, 320-150:320+150]
        # print(image.shape) # (220, 300)
        # normalizing image
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        # adding noise to simulated depth image
        image = image + np.random.normal(0.0, 10.0, image.shape)
        # round to integer values and don't allow values <0 or >255 (necessary because of the added noise)
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)
        if(setRandomPixelsToZero):
            # set approx. 5% of all pixels to zero
            mask = (np.random.uniform(0,1,size=image.shape) > 0.95).astype(np.bool)
            image[mask] = 0
        if(saveImage):
            # saving depth image
            if(saveToFile):
                np.set_printoptions(threshold="nan")
                f = open(folder + "depth_image_sim_" + str(self.imageCount) + ".txt", "w")
                print >>f, image
                f.close()
                # print("depth image saved as file")
            pathToImage = folder + "depth_image_sim_" + str(self.imageCount) + ".jpg"
            self.imageCount += 1
            try:
                cv2.imwrite(pathToImage, image)
                # print("depth image saved as jpg")
            except Exception as e:
                print(e)
        if(self.flattenImage):
            return image.flatten()
        else:
            return image
    # read_kinect_depth_image
    
    def insertObject(self, x, y, z, path_to_sdf, name, roll=0.0, pitch=0.0, yaw=0.0, namespace="world"):
        """ Inserts an object into the gazebo simulation from a SDF file specified by the path. """
        initial_pose = self.createPose(x, y, z, roll, pitch, yaw)
        f = open(path_to_sdf, "r")
        sdff = f.read()
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        spawn_model_prox = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        spawn_model_prox(name, sdff, "", initial_pose, namespace)
    # insertObject

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

    def setPoseOfObjectToPickUp(self, pose):
        """ Sets the pose (position & orientation) of the object to pick up in the gazebo simulation (with zero velocity aka twist). """
        # sets pose and twist of a link.  All children link poses/twists of the URDF tree are not updated accordingly, but should be.
        newLinkState = LinkState()
        # link name, link_names are in gazebo scoped name notation, [model_name::body_name]
        newLinkState.link_name = "objectToPickUp::objectToPickUp_link"
        newLinkState.pose = pose  # desired pose in reference frame
        newTwist = Twist()
        newTwist.linear.x = 0.0
        newTwist.linear.y = 0.0
        newTwist.linear.z = 0.0
        newTwist.angular.x = 0.0
        newTwist.angular.y = 0.0
        newTwist.angular.z = 0.0
        newLinkState.twist = newTwist  # desired twist in reference frame
        # set pose/twist relative to the frame of this link/body, leave empty or "world" or "map" defaults to world-frame
        newLinkState.reference_frame = "world"
        self.link_state_pub.publish(newLinkState)
    # setPoseOfObjectToPickUp

    def getPoseOfObjectAndGripper(self):
        """ Retrieves the pose (position & orientation) of the object to pick up
            and the positions of the two gripper fingers of the robot arm. """
        # print("getting pose (position & orientation) of object to pick up")
        exceptionCount = 0
        object_positions = None
        while object_positions is None:
            try:
                object_positions = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5)
            except Exception as e:
                print("failed to receive position of object to pick up")
                print(e)
                exceptionCount += 1
                if(exceptionCount >= 10):
                    # the Gazebo server crashed
                    print("well this is awkward")
                    print(psutil.cpu_percent())
                    print(psutil.virtual_memory())
                    raise RuntimeError()
        # object_positions = LinkStates() # uncomment this for autocompletion when writing code (Visual Studio Code: Ctrl+Shift+7)
        # print(object_positions.name)
        # ['ground_plane::link', 'pincher::arm_base_link', 'pincher::arm_shoulder_pan_link', 'pincher::arm_shoulder_lift_link',
        # 'pincher::arm_elbow_flex_link', 'pincher::arm_wrist_flex_link', 'tablePlane::link', 'kinect_ros::link', 'objectToPickUp::objectToPickUp_link']
        # ModelStates from topic /gazebo/model_states (not really useful for this task):
        # ['ground_plane', 'tablePlane', 'kinect_ros', 'objectToPickUp', 'pincher']
        indices = [i for i, s in enumerate(object_positions.name) if "objectToPickUp" in s]
        indexOfObjectToPickUp = indices[0]
        pickup_pose = object_positions.pose[indexOfObjectToPickUp]
        # print(pickup_pose)
        # position: 
        #   x: 0.1
        #   y: 6.96718167117e-11
        #   z: 0.0400000000015
        # orientation: 
        #   x: -6.04978974995e-11
        #   y: 1.05331015003e-16
        #   z: 2.20384226181e-12
        #   w: 1.0

        # print("getting position of both gripper finger links (gripper_active_link & gripper_active2_link)")
        # We can only get the arm_wrist_flex_link via the /gazebo/link_states topic, so we get the pose of this link
        # (which doesn't move relative to the finger links) and use this arm_wrist_flex_link pose to calculate
        # the position of the two finger links
        indices = [i for i, s in enumerate(object_positions.name) if "arm_wrist_flex_link" in s]
        indexOfGripper = indices[0]
        arm_wrist_pose = object_positions.pose[indexOfGripper]

        # Getting translations from arm_wrist_flex_link to gripper_active link (we don't care for rotation,
        # as we are only interested in the position of the gripper_active link (and not its orientation as well)).
        # As mentioned above, these two coordinate systems don't move relative to each other.
        listener = tf.TransformListener()
        try:
            listener.waitForTransform("/arm_wrist_flex_link", "/gripper_active_link", rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = listener.lookupTransform("/arm_wrist_flex_link", "/gripper_active_link", rospy.Time())
            # print trans, rot # [8.093486894116319e-08, 0.014999924372865158, 0.08350001990173947] [-9.381838015211219e-07, 0.7071063120916903, 1.2448494643151295e-12, -0.707107250280471]
        except (tf.LookupException, tf.ConnectivityException) as e:
            print("failed to get gripper_active_link position")
            print(e)

        # calculating the unit vectors (described in the /world coordinate system) of the arm_wrist_flex_link
        ex, ey, ez = self.getUnitVectorsFromOrientation(arm_wrist_pose.orientation)
        
        # calculating the positions of the finger links (described in the world coordinate system)
        # position of gripper_active_link (gripper_right_position)
        gripper_right_position = np.matrix([[arm_wrist_pose.position.x],[arm_wrist_pose.position.y],[arm_wrist_pose.position.z]])
        gripper_right_position = gripper_right_position + trans[0]*ex + trans[1]*ey + trans[2]*ez
        # print(gripper_right_position) # e.g. [[0.09] [0.013] [0.058]]

        # position of gripper_active2_link (gripper_left_position)
        gripper_left_position = np.matrix([[arm_wrist_pose.position.x],[arm_wrist_pose.position.y],[arm_wrist_pose.position.z]])
        gripper_left_position = gripper_left_position + trans[0]*ex - trans[1]*ey + trans[2]*ez
        # print(gripper_left_position) # e.g. [[0.09] [-0.013] [0.058]]

        return pickup_pose, gripper_right_position, gripper_left_position
    # getPoseOfObjectAndGripper

    def getUnitVectorsFromOrientation(self, quaternion):
        """ Calculates the unit vectors (described in the /world coordinate system)
            of an object, which orientation is given by the quaternion. """
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(explicit_quat)

        rot_mat = tf.transformations.euler_matrix(roll, pitch, yaw)
        ex = np.dot(rot_mat, np.matrix([[1], [0], [0], [1]])) # e.g. [[0.800] [0.477] [-0.362] [1.]]
        ex = ex[:3] / ex[3]

        ey = np.dot(rot_mat, np.matrix([[0], [1], [0], [1]]))
        ey = ey[:3] / ey[3]

        ez = np.dot(rot_mat, np.matrix([[0], [0], [1], [1]]))
        ez = ez[:3] / ez[3]
        return ex, ey, ez
    # getUnitVectorsFromOrientation

    def calculateReward(self, pickup_pose_old, pickup_pose, gripper_right_position, gripper_left_position):
        """ Calculates the reward for the current timestep, according to the gripper position and the pickup position. 
            A high reward is given if the gripper could grasp the box (pickup) if it would close the gripper. """
        pickup_position_old = np.matrix([[pickup_pose_old.position.x],[pickup_pose_old.position.y],[pickup_pose_old.position.z]])
        
        pickup_position = np.matrix([[pickup_pose.position.x],[pickup_pose.position.y],[pickup_pose.position.z]])
        # print("pickup_position:")
        # print(pickup_position) # e.g. [[0.1], [0.0], [0.04]]

        # check if the gripper has crashed into the ObjectToPickUp
        tolerance = 0.005
        if(pickup_position[0] < pickup_position_old[0] - tolerance or pickup_position[0] > pickup_position_old[0] + tolerance or
            pickup_position[1] < pickup_position_old[1] - tolerance or pickup_position[1] > pickup_position_old[1] + tolerance or 
            pickup_position[2] < pickup_position_old[2] - tolerance or pickup_position[2] > pickup_position_old[2] + tolerance):
            pass
            # print("********************************************* gripper crashed into the ObjectToPickUp! *********************************************")
            # return self.rewardFailure
        
        # check if gripper is in the correct position in order to grasp the object
        
        # dimensions of the ObjectToPickUp & the gripper (see the corresponding sdf/urdf files)
        pickup_xdim = 0.07
        pickup_ydim = 0.02
        pickup_zdim = 0.03
        gripper_width = 0.032 # between the fingers
        gripper_height = 0.035

        # calculating the unit vectors (described in the /world coordinate system) of the objectToPickUp::link
        ex, ey, ez = self.getUnitVectorsFromOrientation(pickup_pose.orientation)
        
        # corners of bounding box where gripper_right_position has to be
        p1 = pickup_position + pickup_xdim/2 * ex + pickup_ydim/2 * ey
        p2 = p1 - pickup_xdim * ex
        p4 = p1 + ey * (pickup_ydim/2 + gripper_width - pickup_ydim)
        p5 = p1 + ez * pickup_zdim
        # print("corners of bounding box where gripper_right_position has to be:")
        # print(p1) # e.g. [[0.135] [0.01 ] [0.04]]
        # print(p2) # e.g. [[0.065] [0.01 ] [0.04]]
        # print(p4) # e.g. [[0.135] [0.032] [0.04]]
        # print(p5) # e.g. [[0.135] [0.01 ] [0.07]]

        # the vectors of the bounding box where gripper_right_position has to be
        # transpose is necessary because np.dot can't handle two (3,1) matrices, so one of them has to be a (1,3) matrix
        u = np.transpose(p2 - p1) # e.g. [[ -7.00000000e-02  1.38777878e-17 -2.77555756e-17]]
        v = np.transpose(p4 - p1)
        w = np.transpose(p5 - p1)

        # corners of bounding box where gripper_left_position has to be
        p1_left = pickup_position + pickup_xdim/2 * ex - pickup_ydim/2 * ey
        p2_left = p1_left - pickup_xdim * ex
        p4_left = p1_left - ey * (pickup_ydim/2 + gripper_width - pickup_ydim)
        p5_left = p1_left + ez * pickup_zdim
        # print("corners of bounding box where gripper_left_position has to be:")
        # print(p1_left) # e.g. [[0.135] [-0.01 ] [0.04]]
        # print(p2_left) # e.g. [[0.065] [-0.01 ] [0.04]]
        # print(p4_left) # e.g. [[0.135] [-0.032] [0.04]]
        # print(p5_left) # e.g. [[0.135] [-0.01 ] [0.07]]

        # the vectors of the bounding box where gripper_left_position has to be
        u_left = np.transpose(p2_left - p1_left)
        v_left = np.transpose(p4_left - p1_left)
        w_left = np.transpose(p5_left - p1_left)

        graspSuccess = False
        # check if right gripper is on the right and left gripper is on the left of the ObjectToPickUp
        gripperRightIsRight = self.isPositionInCuboid(gripper_right_position, p1, p2, p4, p5, u, v, w)
        gripperLeftIsLeft = self.isPositionInCuboid(gripper_left_position, p1_left, p2_left, p4_left, p5_left, u_left, v_left, w_left)
        # check if right gripper is on the left and left gripper is on the right of the ObjectToPickUp
        gripperLeftIsRight = self.isPositionInCuboid(gripper_left_position, p1, p2, p4, p5, u, v, w)
        gripperRightIsLeft = self.isPositionInCuboid(gripper_right_position, p1_left, p2_left, p4_left, p5_left, u_left, v_left, w_left)
        # if one of the two scenarios is true, the grasping would be successful
        if((gripperRightIsRight and gripperLeftIsLeft) or (gripperLeftIsRight and gripperRightIsLeft)):
            print("********************************************* grasping would be successful! *********************************************")
            graspSuccess = True
        else:
            graspSuccess = False
        
        if(self.binaryReward):
            if(graspSuccess):
                return self.rewardSuccess
            else:
                return self.rewardFailure
        else:
            # calculate reward according to the distance from the gripper to the middle of the bounding box
            # pM = middle of the box where gripper_right_position has to be
            pM = p1 + 0.5 * np.transpose(u) + 0.5 * np.transpose(v) + 0.5 * np.transpose(w) # e.g. matrix([[0.1],[0.021],[0.055]])
            distance = np.linalg.norm(pM - gripper_right_position) # e.g. 0.1375784482561938
            # invert the distance, because smaller distance == closer to the goal == more reward
            reward = 1.0 / distance # e.g. 7.2685803094525
            # scale the reward if gripper is in the bounding box
            # (if gripper_right_position is exaclty at an edge of bounding box (e.g. p1), unscaled reward would be approx 25.2)
            if(graspSuccess):
                reward = 5 * reward
            # print("received reward: " + str(reward))
            return reward
        # if
    # calculateReward

    def isPositionInCuboid(self, gripper_position, p1, p2, p4, p5, u, v, w):
        """ Checks if gripper_position is in the correct location (i.e. within the cuboid described by u, v & w). """
        # (see https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d)
        if(np.dot(u, gripper_position) > np.dot(u, p1) and np.dot(u, gripper_position) < np.dot(u, p2)):
            # print("gripper is in correct position (x-axis)")
            if(np.dot(v, gripper_position) > np.dot(v, p1) and np.dot(v, gripper_position) < np.dot(v, p4)):
                # print("gripper is in correct position (y-axis)")
                if(np.dot(w, gripper_position) > np.dot(w, p1) and np.dot(w, gripper_position) < np.dot(w, p5)):
                    # print("gripper is in correct position (z-axis)")
                    return True
        return False
    # isPositionInCuboid

    def moveArmToHomePosition(self):
        """ Moves the arm to its home position. """
        # Option 1: setting all joint values to zero (if that's not already the case)
        numTries = 0
        while(not self.armIsInHomePosition()):
            if(numTries > 0):
                print("move arm to home position try nr. {}".format(numTries + 1))
                rospy.sleep(0.2)
            numTries += 1
            self.group_arm.clear_pose_targets()
            arm_joint_values = self.group_arm.get_current_joint_values()
            for i in range(len(arm_joint_values)):
                arm_joint_values[i] = 0.0
            self.group_arm.set_joint_value_target(arm_joint_values)
            traj = self.group_arm.plan()
            success = self.executeTrajectory(traj, speedUp=3.0)
        # while
        
        return True
        
        # Option 2: named pose (defined in the moveit package of the robot arm)
        # self.group_arm.clear_pose_targets()
        # self.group_arm.set_named_target("right_up")
        # planHome = self.group_arm.plan()
        # successTry1 = self.group_arm.execute(planHome)
        # successTry1 = self.group_arm.go(wait=True)
    # moveArmToHomePosition

    def moveArmToPose(self, targetPose, execute=True):
        """ Moves the arm to the specified pose. """
        self.group_arm.clear_pose_targets()
        self.group_arm.set_pose_target(targetPose)
        # plan the trajectory
        traj = self.group_arm.plan()
        couldFindPlan = len(traj.joint_trajectory.points) > 0
        if(execute):
            success = self.executeTrajectory(traj, speedUp=3.0)
            # Sometimes success == False, even though the arm has moved! So we have to check manually if the arm has moved
            armHasMoved = not self.armIsInHomePosition()
            success = success or armHasMoved
            if(couldFindPlan and not success):
                print("********************************************* plan found but failed to execute! *********************************************")
            return success
        else:
            return couldFindPlan
    # moveArmToPose

    def executeTrajectory(self, traj, speedUp=1.0):
        """ Executes the given trajectory with a possible speed up to accelerate the learning. """
        # Taken from: https://github.com/ros-planning/moveit_ros/issues/368#issuecomment-29717359
        new_traj = RobotTrajectory()
        new_traj = traj

        n_joints = len(traj.joint_trajectory.joint_names)
        n_points = len(traj.joint_trajectory.points)

        points = list(traj.joint_trajectory.points)

        for i in range(n_points):  
            point = JointTrajectoryPoint()
            point.time_from_start = traj.joint_trajectory.points[i].time_from_start / speedUp
            point.velocities = list(traj.joint_trajectory.points[i].velocities)
            point.accelerations = list(traj.joint_trajectory.points[i].accelerations)
            point.positions = traj.joint_trajectory.points[i].positions

            for j in range(n_joints):
                point.velocities[j] = point.velocities[j] * speedUp
                point.accelerations[j] = point.accelerations[j] * speedUp**2

            points[i] = point

        new_traj.joint_trajectory.points = points

        # execute the trajectory
        success = self.group_arm.execute(new_traj, wait=True)
        return success
    # executeTrajectory

    def armIsInHomePosition(self):
        """ Returns True if arm is in its home position (i.e. all joint values are 0) """
        arm_joint_values = self.group_arm.get_current_joint_values()
        return all(abs(joint_value) < 0.001 for joint_value in arm_joint_values)
    # armIsInHomePosition
 
    def printMovingArmSuccess(self, success, printFailure=False, printSuccess=False, target=""):
        if(not success and printFailure):
            print("********************************************* FAILED TO MOVE ARM " + target + " *********************************************")
        elif(success and printSuccess):
            print("********************************************* MOVED ARM SUCCESSFULLY " + target + " *********************************************")
    # printMovingArmSuccess

# class GazeboSmartBotPincherKinectEnv
