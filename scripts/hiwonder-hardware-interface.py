from math import *
import copy
import traceback
import time
import numpy as np

from funrobo_hiwonder.core.hiwonder import HiwonderRobot

import funrobo_kinematics.core.utils as ut
from hiwonder import Hiwonder


SQUARE_POSE = [
    np.array([0.060, 0.38, 0]),
    np.array([-0.060, 0.38, 0]),
    np.array([-0.060, 0.240, -0.05]),
    np.array([0.060, 0.240, -0.05]),
]

STAR_POSE = [
    np.array([0, 0.38, 0]), # 1

    np.array([0.050, 0.240, -0.05]), #4

    np.array([-0.060, 0.38-.045, -.015]), #2

    np.array([0.060, 0.38-.045, -.015]), #5

    np.array([-0.050, 0.240, -0.05]), #3
      
]

K_WORD_POSE = [
    np.array([-0.045, 0.38-.035, 0]), #1
    np.array([-0.045, 0.24+.028, -.04]), #2
    np.array([0.04, 0.38-.03, 0]), #3
    np.array([-0.045, 0.38-0.075, -.02]), #4
    np.array([0.04, 0.24+.035, -.04]), #5
]

E_WORD_POSE = [
    np.array([-0.045, 0.38-.035, 0]), #1
    np.array([-0.045, 0.24+.021, -.04]), #2    
    np.array([0.04, 0.38-.03, 0]), #3
    np.array([-0.045, 0.38-.035, 0]), #4
    np.array([0.03, 0.38-0.08, -.02]), #5
    np.array([-0.045, 0.38-0.08, -.02]), #6
    np.array([0.04, 0.24+.021, -.04]), #7
    np.array([-0.045, 0.24+.021, -.04]), #8
]

N_WORD_POSE = [

    np.array([-0.045, 0.24+.018, -.045]), #1  
    np.array([-0.045, 0.38-.035, -0.015]), #2
    np.array([0.04, 0.24+.021, -.04]), #3
    np.array([0.04, 0.38-.03, 0]), #4




]

Ry_0T = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
Rz_0T = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
R_0T = Ry_0T @ Rz_0T
d_0T = np.array([[-0.22], [0], [0]])


def precompute_ik(model, poses_task):
    """Transform task-frame poses to robot frame, then solve IK for each."""
    poses_robot = [pose @ R_0T.T + d_0T.T for pose in poses_task]

    initial_guess = [0.0, 0.0, pi / 2, -pi / 6, 0.0]
    joint_angles_list = []

    prev_solution = initial_guess
    for i, pose in enumerate(poses_robot):
        p_ee = ut.EndEffector()
        p_ee.x, p_ee.y, p_ee.z = pose.flatten()
        joint_vals = model.calc_numerical_ik(p_ee, initial_guess=copy.deepcopy(prev_solution))
        joint_angles_list.append(copy.deepcopy(joint_vals))
        prev_solution = joint_vals
        print(f"  Pose {i}: task={poses_task[i]} -> robot={pose.flatten()} -> joints(deg)={np.rad2deg(joint_vals)}")

    return joint_angles_list


def run_interface(model, joint_angles_list):
    """Main loop: RRMC joystick control + IK pose cycling on home press."""
    try:
        robot = HiwonderRobot()

        control_hz = 30
        dt = 1 / control_hz

        print("Ready. Use joystick for RRMC, home button to cycle poses.")

        pose_idx = 0
        home_move_until = 0.0
        while True:
            t_start = time.time()

            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break

            if robot.gamepad.cmdlist:
                cmd = robot.gamepad.cmdlist[-1]

                if cmd.arm_home:
                    time.sleep(0.2)

                    target_joints_rad = joint_angles_list[pose_idx % len(joint_angles_list)]
                    target_joints_deg = np.rad2deg(target_joints_rad).tolist()
                    all_joints_deg = target_joints_deg + [0.0]

                    print(f"[INFO] Home -> pose {pose_idx % len(joint_angles_list)}: {all_joints_deg}")
                    robot.set_joint_values(all_joints_deg, duration=1.0, radians=False)
                    pose_idx += 1
                    home_move_until = time.time() + 1.0
                    continue

                if time.time() < home_move_until:
                    continue

                curr_joint_values = robot.get_joint_values()

                speed = 1.0
                vel = [speed * cmd.arm_vx, speed * cmd.arm_vy, speed * cmd.arm_vz]
                arm_joints_deg = curr_joint_values[:5]
                arm_joints_rad = np.deg2rad(arm_joints_deg).tolist()

                new_arm_joints_rad = model.calc_velocity_kinematics(arm_joints_rad, vel, dt=dt)

                new_arm_joints_deg = np.rad2deg(new_arm_joints_rad).tolist()
                new_joint_values = new_arm_joints_deg + [curr_joint_values[5]]
                robot.set_joint_values(new_joint_values, duration=dt, radians=False)

            elapsed = time.time() - t_start
            remaining_time = dt - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        robot.shutdown_robot()


if __name__ == "__main__":

    model = Hiwonder()

    ik_poses = precompute_ik(model, N_WORD_POSE)
    print(f"Precomputed {len(ik_poses)} poses. Starting interface...")

    run_interface(model, ik_poses)
