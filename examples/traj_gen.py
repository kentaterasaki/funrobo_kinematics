
import numpy as np

from funrobo_kinematics.core.trajectory_generator import (
    CubicPolynomial,
    QuinticPolynomial,
    Trapezoidal,
    MultiAxisTrajectoryGenerator,
    MultiSegmentTrajectoryGenerator,
)


def main():
    ndof = 2
    method = Trapezoidal(ndof=ndof)
    mode = "joint"
    type = 1

    # --------------------------------------------------------
    # Point-to-point multi-axis trajectory generator
    # --------------------------------------------------------
    if type == 1:
        traj = MultiAxisTrajectoryGenerator(method=method,
                                            mode=mode,
                                            ndof=ndof)
        
        traj.solve(q0=[-30, -30], qf=[60, 60], T=1)
        traj.generate(nsteps=20)

    # --------------------------------------------------------
    # Via point multi-axis trajectory generator
    # --------------------------------------------------------
    if type == 2:
        traj = MultiSegmentTrajectoryGenerator(method=method,
                                            mode=mode,
                                            ndof=ndof,
                                                )
        via_points = [[-30, 30], [0, 45], [30, 15], [50, -30]]

        traj.solve(via_points, T=2)
        traj.generate(nsteps_per_segment=20)
    
    
    # plotter
    traj.plot()

if __name__ == "__main__":
    main()