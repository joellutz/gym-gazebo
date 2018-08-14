import numpy as np
import tf


def main():


    pickup_xdim = 0.07
    pickup_ydim = 0.02
    pickup_zdim = 0.03

    # calculating the unit vectors (described in the /world coordinate system) of the objectToPickUp::link
    # Option 1: using numpy algebra
    rot_mat = tf.transformations.euler_matrix(0.0, 0.0, 0.0)
    # print(rot_mat)
    # rot_mat_x = np.matrix([[1,0,0], [0,np.cos(roll), -np.sin(roll)], [0,np.sin(roll), np.cos(roll)]])
    # print(rot_mat_x)
    ex = np.dot(rot_mat, np.matrix([[1], [0], [0], [1]]))
    # print(ex)
    # [[ 0.80017017]
    # [ 0.4778494 ]
    # [-0.36247434]
    # [ 1.        ]]
    ex = ex[:3] / ex[3]
    print(ex)
    print(np.linalg.norm(ex))
    print(type(ex))

    ey = np.dot(rot_mat, np.matrix([[0], [1], [0], [1]]))
    ey = ey[:3] / ey[3]

    ez = np.dot(rot_mat, np.matrix([[0], [0], [1], [1]]))
    ez = ez[:3] / ez[3]

    gripper_pose = np.matrix([[-3.0627457634e-06], [0.0], [0.438999798398]])
    pickup_pose = np.matrix([[0.1], [0.1], [0.06742099]])

    print("gripper_pose:")
    print(gripper_pose)
    print(type(gripper_pose))
    print("pickup_pose:")
    print(pickup_pose)

    p1 = pickup_pose + pickup_xdim/2 * ex + ey * pickup_ydim/2 - ez * pickup_zdim/2
    p2 = p1 + pickup_xdim * ex
    p4 = p1 + ey * pickup_ydim
    p5 = p1 + ez * pickup_zdim

    u = np.transpose(p1 - p2)
    v = np.transpose(p1 - p4)
    w = np.transpose(p1 - p5)

    print("u:")
    print(u)
    print(type(u))
    print("v:")
    print(v)
    print("w:")
    print(w)

    print("u * gripper_pose:")
    print(np.shape(u))
    print(np.shape(gripper_pose))
    print(np.dot(u, gripper_pose))
    print(type(np.dot(u, gripper_pose)))
    print("u * p1:")
    print(np.dot(u, p1))
    print("u * p2:")
    print(np.dot(u, p2))

    if(np.dot(u, gripper_pose) > np.dot(u, p1) and np.dot(u, gripper_pose) < np.dot(u, p2)):
        print("gripper is in correct position (x-axis)")
    if(np.dot(v, gripper_pose) > np.dot(v, p1) and np.dot(v, gripper_pose) < np.dot(v, p4)):
        print("gripper is in correct position (y-axis)")
    if(np.dot(w, gripper_pose) > np.dot(w, p1) and np.dot(w, gripper_pose) < np.dot(w, p5)):
        print("gripper is in correct position (z-axis)")


if __name__ == '__main__':
    main()
