from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate


class Hiwonder(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()

        th1, th2, th3, th4, th5 = curr_joint_values
        l1, l2, l3, l4, l5 = (
            self.l1,
            self.l2,
            self.l3,
            self.l4,
            self.l5,
        )
        dh_01 = [th1, l1, 0, -(pi / 2)]
        dh_12 = [th2 - (pi / 2), 0, l2, pi]
        dh_23 = [th3, 0, l3, pi]
        dh_34 = [(pi / 2) + th4, 0, 0, (pi / 2)]
        dh_45 = [th5, l4 + l5, 0, 0]
        dh_tables = [dh_01, dh_12, dh_23, dh_34, dh_45]
        H_matrix = np.eye(4)
        self.Hlist = []

        for table in dh_tables:
            H_matrix_new = ut.dh_to_matrix(table)
            H_matrix = H_matrix @ H_matrix_new
            self.Hlist.append(H_matrix_new)

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_matrix @ np.array([0, 0, 0, 1]))[:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_matrix[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, self.Hlist

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.01):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """
        new_joint_values = joint_values.copy()

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [
                theta + np.random.rand() * 0.02 for theta in new_joint_values
            ]

        # Calculate joint velocities using the inverse Jacobian
        joint_vel = self.inverse_jacobian(new_joint_values) @ vel

        joint_vel = np.clip(
            joint_vel,
            [limit[0] for limit in self.joint_vel_limits],
            [limit[1] for limit in self.joint_vel_limits],
        )

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(
            new_joint_values,
            [limit[0] for limit in self.joint_limits],
            [limit[1] for limit in self.joint_limits],
        )

        return new_joint_values

    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the robot.

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            np.ndarray: The Jacobian matrix (3x5).
        """

        ee, Hlist = self.calc_forward_kinematics(joint_values)

        jacobian = np.zeros((3, 5))
        H_cumulative = np.eye(4)

        r_ee = np.array([ee.x, ee.y, ee.z])

        for i in range(5):
            z_i = H_cumulative[:3, 2]
            r_i = H_cumulative[:3, 3]

            jacobian[:, i] = np.cross(z_i, r_ee - r_i)
            H_cumulative = H_cumulative @ Hlist[i]
        return jacobian

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.jacobian(joint_values))

    def calc_numerical_ik(self, ee, initial_guess=None, tol=0.002, ilimit=100):
        mins = np.array([l[0] for l in self.joint_limits])
        maxs = np.array([l[1] for l in self.joint_limits])

        for i in range(ilimit):
            if initial_guess is not None and i == 0:
                guess = np.array(initial_guess, dtype=float)
            else:
                guess = ut.sample_valid_joints(self, ilimit)
            curr_joint_values = guess
            for j in range(ilimit):
                curr_ee_obj, _ = self.calc_forward_kinematics(curr_joint_values)

                diff = np.array(
                    [ee.x - curr_ee_obj.x, ee.y - curr_ee_obj.y, ee.z - curr_ee_obj.z]
                )

                error_norm = np.linalg.norm(diff)
                if error_norm < tol:
                    print("Converged")
                    return curr_joint_values.tolist()

                step = self.inverse_jacobian(curr_joint_values) @ diff
                curr_joint_values += step

                curr_joint_values = np.clip(curr_joint_values, mins, maxs)

        print("IK failed to converge")
        return curr_joint_values.tolist()

    def calc_inverse_kinematics(self, ee: ut.EndEffector, joint_values, soln=0):
        """
        Compute an analytical inverse kinematics solution.

        Given a desired end-effector pose, this method computes a set of joint
        angles that achieve the pose, if a valid solution exists.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (List[float]): Initial or previous joint angles, used
                for solution selection or continuity.
            soln (int, optional): Solution branch index (e.g., elbow-up vs
                elbow-down). Defaults to 0.

        Returns:
            list[float]: Joint angles [theta1, theta2] in radians that achieve
            the desired end-effector pose.
        """
        l1, l2, l3, l4, l5 = (
            self.l1,
            self.l2,
            self.l3,
            self.l4,
            self.l5,
        )
        position = [ee.x, ee.y, ee.z]
        euler = (ee.rotx, ee.roty, ee.rotz)
        R05 = ut.euler_to_rotm(euler)
        zR05 = R05 @ [0, 0, 1]
        new_joint_list = []
        p4 = position - zR05 * (l4 + l5)
        for sol_idx in range(4):
            # Shoulder: soln 0,1 = front; soln 2,3 = back
            th1_front = atan2(p4[1], p4[0])
            th1 = th1_front if sol_idx < 2 else th1_front - pi
            plane = np.linalg.norm([p4[0], p4[1]])
            z_rel = p4[2] - l1

            # For back-shoulder, flip the plane projection
            if sol_idx >= 2:
                plane = -plane

            L_sq = plane**2 + z_rel**2
            beta = acos((l2**2 + l3**2 - L_sq) / (2 * l2 * l3))

            # Elbow: soln 0,2 = elbow-up; soln 1,3 = elbow-down
            th3 = (beta - pi) if sol_idx % 2 == 0 else (pi - beta)
            alpha = atan2(l3 * sin(th3), l2 + l3 * cos(th3))
            gamma = atan2(plane, z_rel)
            th2 = gamma + alpha

            dh_tables = np.array(
                [
                    [th1, l1, 0, -(pi / 2)],
                    [th2 - (pi / 2), 0, l2, pi],
                    [th3, 0, l3, pi],
                ]
            )
            R03 = np.eye(3)
            for table in dh_tables:
                theta = table[0]
                a = table[3]
                T = np.array(
                    [
                        [cos(theta), -sin(theta) * cos(a), sin(theta) * sin(a)],
                        [sin(theta), cos(theta) * cos(a), -cos(theta) * sin(a)],
                        [0, sin(a), cos(a)],
                    ]
                )
                R03 = R03 @ T
            R35 = R03.T @ R05
            th4 = atan2(R35[1, 2], R35[0, 2])
            th5 = atan2(R35[2, 0], R35[2, 1])
            # Wrap angles to [-pi, pi] for removing illegal joint angles
            wrap = lambda a: (a + pi) % (2 * pi) - pi
            new_joint_values = [wrap(th1), wrap(th2), wrap(th3), wrap(th4), wrap(th5)]
            # new_joint_values = [th1, th2, th3, th4, th5]
            if ut.check_valid_ik_soln(new_joint_values, ee, self):
                new_joint_list.append(new_joint_values)
        print(new_joint_list)
        joint_value = new_joint_list[soln]

        return joint_value

    def plan_path(self, joint_values, desired_ee_list):
        curr_ee = self.calc_forward_kinematics(joint_values)
        x_path = []
        y_path = []
        z_path = []
        curr_x, curr_y, curr_z = curr_ee[0], curr_ee[1], curr_ee[2]

        for ee in desired_ee_list:
            x_path.extend(np.linspace(curr_x, ee[0]))
            y_path.extend(np.linspace(curr_y, ee[1]))
            z_path.extend(np.linspace(curr_z, ee[2]))
            curr_x, curr_y, curr_z = ee[0], ee[1], ee[2]
        path = np.column_stack((x_path, y_path, z_path))
        joint_val_list = []
        for coords in path:
            joint_val_list.append(self.calc_numerical_ik(coords, 0))
        return joint_val_list


if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = Hiwonder()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
