from __future__ import division
from os import link
import sim_update
import pybullet as p
import random
import numpy as np
import math
import matplotlib.pyplot as plt

MAX_ITERS = 10000
delta_q = 0.1

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env, distance=0.12):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :param distance: threshold distance for check_collision
    :returns path: series of joint angles
    """
    # ========== PART 3 =========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    V, E = [q_init], []
    path, found = [], False

    for i in range(MAX_ITERS):
        q_rand = semi_random_sample(steer_goal_p, q_goal)
        q_nearest = nearest(V, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)
        if not env.check_collision(q_new, 0.12):
            if q_new not in V:
                V.append(q_new)
            if (q_nearest, q_new) not in E:
                E.append((q_nearest, q_new))
                visualize_path(q_nearest, q_new, env)
            if get_euclidean_distance(q_goal, q_new) < delta_q:
                V.append(q_goal)
                E.append((q_new, q_goal))
                visualize_path(q_new, q_goal, env)
                found = True
                break

    if found:
        current_q = q_goal
        path.append(current_q)
        while current_q != q_init:
            for edge in E:
                if edge[1] == current_q:
                    current_q = edge[0]
                    path.append(edge[0])
        path.reverse()
        return path
    else:
        return None


def semi_random_sample(steer_goal_p, q_goal):
    """
    :param steer_goal_p: probability of steering towards the goal
    :param q_goal: goal configuration

    :returns q_rand: a uniform random sample in free Cspace
    """
    prob = random.random()

    if prob < steer_goal_p:
        return q_goal
    else:
        # Uniform sample over reachable joint angles
        q_rand = [random.uniform(-np.pi, np.pi) for i in range(len(q_goal))]
    return q_rand


def get_euclidean_distance(q1, q2):
    distance = 0
    for i in range(len(q1)):
        distance += (q2[i] - q1[i])**2
    return math.sqrt(distance)


def nearest(V, q_rand):
    """
    :param V: vertices in the current tree
    :param q_rand: a new uniform random sample

    :returns q_nearest: the closet point on the tree to q_rand
    """
    # Using euclidean distance
    distance = float("inf")
    q_nearest = None
    for idx, v in enumerate(V):
        if get_euclidean_distance(q_rand, v) < distance:
            q_nearest = v
            distance = get_euclidean_distance(q_rand, v)
    return q_nearest


def steer(q_nearest, q_rand, delta_q):
    q_new = None
    if get_euclidean_distance(q_rand, q_nearest) <= delta_q:
        q_new = q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / get_euclidean_distance(q_rand, q_nearest) for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
    return q_new


def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    # ========= PART 2============
    # Get position and orientation (yaw in radians) of the gripper for grasping
    # ==================================
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle


if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim_update.PyBulletSim(object_shapes = object_shapes)
    num_trials = 3  # Tăng số lần thử nghiệm lên 50
    
    # Danh sách lưu trữ độ dài đường đi của từng lần chạy
    path_lengths = []

    # PART 3: RRT Implementation
    env.load_gripper()
    passed = 0
    for trial in range(num_trials):
        print(f"Thử nghiệm thứ: {trial + 1}")
        # grasp the object
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            # get a list of robot configuration in small step sizes
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print("No collision-free path is found within the time budget. Continuing...")
                path_lengths.append(None)  # Ghi nhận không tìm thấy đường đi
            else:
                # Tính độ dài đường đi
                path_length = 0
                for i in range(1, len(path_conf)):
                    path_length += get_euclidean_distance(path_conf[i-1], path_conf[i])
                path_lengths.append(path_length)
                
                # Execute the path while visualizing the location of joint 5
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.05)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))

                print("Path executed. Dropping the object")

                # Drop the object
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                # Retrace the path to original location
                for joint_state in reversed(path_conf):
                    env.move_joints(joint_state, speed=0.1)
                markers = None
            p.removeAllUserDebugItems()

        env.robot_go_home()

        # Kiểm tra xem đối tượng có được chuyển đến thùng thứ hai hay không
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()