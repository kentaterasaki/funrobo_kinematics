from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    KinovaRobotTemplate,
)


class KinovaRobot(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()

        th1, th2, th3, th4, th5, th6 = curr_joint_values
        l1, l2, l3, l4, l5, l6, l7 = (
            self.l1,
            self.l2,
            self.l3,
            self.l4,
            self.l5,
            self.l6,
            self.l7,
        )

        dh_b0 = [0, 0, 0, np.pi]
        dh_01 = [th1, -(l1 + l2), 0, (np.pi / 2)]
        dh_12 = [th2 - (np.pi / 2), 0, l3, np.pi]
        dh_23 = [th3 - (np.pi / 2), 0, 0, (np.pi / 2)]
        dh_34 = [th4, -(l4 + l5), 0, -(np.pi / 2)]
        dh_45 = [th5, 0, 0, (np.pi / 2)]
        dh_56 = [th6, -(l6 + l7), 0, np.pi]
        dh_tables = [dh_b0, dh_01, dh_12, dh_23, dh_34, dh_45, dh_56]
        H_matrix = np.eye(4)
        Hlist = []

        for table in dh_tables:
            H_matrix_new = ut.dh_to_matrix(table)
            H_matrix = H_matrix @ H_matrix_new
            Hlist.append(H_matrix_new)

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_matrix @ np.array([0, 0, 0, 1]))[:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_matrix[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist

    def calc_inverse_kinematics(self, ee, joint_values, soln=0):
        l1, l2, l3, l4, l5, l6, l7 = (
            self.l1,
            self.l2,
            self.l3,
            self.l4,
            self.l5,
            self.l6,
            self.l7,
        )
        position = [ee.x, ee.y, ee.z]
        euler = (ee.rotx, ee.roty, ee.rotz)
        R06 = ut.euler_to_rotm(euler)
        zR06 = R06 @ [0, 0, 1]
        p_wrist = position - zR06 * (l6 + l7)
        new_joint_list = []

        d_shoulder = abs(l1 + l2)
        upper_arm = l3
        forearm = abs(l4 + l5)

        for sol_idx in range(4):
            if sol_idx < 2:
                th1 = -atan2(p_wrist[1], p_wrist[0])
                plane = sqrt(p_wrist[0] ** 2 + p_wrist[1] ** 2)
            else:
                th1 = -atan2(-p_wrist[1], -p_wrist[0])
                plane = -sqrt(p_wrist[0] ** 2 + p_wrist[1] ** 2)
            z_rel = p_wrist[2] - d_shoulder
            L_sq = plane**2 + z_rel**2

            ratio = (upper_arm**2 + forearm**2 - L_sq) / (2 * upper_arm * forearm)
            beta = acos(np.clip(ratio, -1.0, 1.0))

            th3 = (beta - pi) if sol_idx % 2 == 0 else (pi - beta)
            alpha = atan2(forearm * sin(th3), upper_arm + forearm * cos(th3))
            psi = atan2(z_rel, plane)
            th2 = ut.wraptopi(pi / 2 - psi + alpha)

            R_B0 = ut.dh_to_matrix([0, 0, 0, pi])[:3, :3]
            R_01 = ut.dh_to_matrix([th1, -(l1 + l2), 0, pi / 2])[:3, :3]
            R_12 = ut.dh_to_matrix([th2 - (pi / 2), 0, l3, pi])[:3, :3]
            R_23 = ut.dh_to_matrix([th3 - (pi / 2), 0, 0, pi / 2])[:3, :3]
            R03 = R_B0 @ R_01 @ R_12 @ R_23

            R36 = R03.T @ R06
            c5 = -R36[2, 2]
            th5_pos = atan2(sqrt(1 - c5**2), c5)
            th5_neg = atan2(-sqrt(1 - c5**2), c5)

            for th5 in [th5_pos, th5_neg]:
                if abs(sin(th5)) > 1e-6:
                    th4 = atan2(R36[1, 2], R36[0, 2])
                    th6 = atan2(R36[2, 1], R36[2, 0])
                else:
                    th4 = 0
                    th6 = atan2(-R36[0, 1], R36[0, 0])

                wrap = lambda a: (a + pi) % (2 * pi) - pi
                new_joint_values = [
                    wrap(th1),
                    wrap(th2),
                    wrap(th3),
                    wrap(th4),
                    wrap(th5),
                    wrap(th6),
                ]

                if ut.check_valid_ik_soln(new_joint_values, ee, self):
                    new_joint_list.append(new_joint_values)

        if not new_joint_list:
            print("IK failed to find a valid solution")
            return joint_values

        joint_value = new_joint_list[min(soln, len(new_joint_list) - 1)]
        return joint_value


if __name__ == "__main__":
    model = KinovaRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
