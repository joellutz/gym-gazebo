#!/usr/bin/python
from subprocess import Popen, call
import sys
import time

trainingCompleted = False
runCount = 0
start_time = time.time()
filename = "/home/joel/Documents/gym-gazebo/examples/pincher_arm/smartbot_pincher_kinect_ddpg.py" #sys.argv[1]

while not trainingCompleted:
    print("\nStarting " + filename)
    start_time_run = time.time()
    runCount += 1
    p = Popen("python " + filename, shell=True)
    code = p.wait()
    print("exit-code of python: {}".format(code))
    print("runtime of attempt {}: {}s".format(runCount, time.time() - start_time_run))
    if(code == 0):
        print("training successfully completed")
        print("total runtime ({} attempts): {}s".format(runCount, time.time() - start_time))
        trainingCompleted = True
    else:
        cmd = "killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient" # aka killgazebogym
        print("killgazebogym & try again")
        result = call(cmd, shell=True)
        time.sleep(10)
    # if
# while
