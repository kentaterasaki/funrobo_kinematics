import numpy as np
import matplotlib.pyplot as plt


class CubicPolynomial():
    """
    Cubic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        self.ndof = ndof

    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)

        A = np.array(
                [[1, t0, t0**2, t0**3],
                 [0, 1, 2*t0, 3*t0**2],
                 [1, tf, tf**2, tf**3],
                 [0, 1, 2*tf, 3*tf**2]
                ])

        b = np.vstack([
            q0,
            qd0,
            qf,
            qdf
        ])
        self.coeff = np.linalg.solve(A, b)

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof):
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2
            qdd = 2 * c[2] + 6 * c[3] * t

            X[i, 0, :] = q
            X[i, 1, :] = qd
            X[i, 2, :] = qdd

        return t, X


class QuinticPolynomial():
    """
    Quintic interpolation with position, velocity, and acceleration boundary constraints.
    """

    def __init__(self, ndof=None):
        self.ndof = ndof

    def solve(self, q0, qf, qd0, qdf, qdd0, qddf, T):
        """
        Compute quintic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        qdd0 : array-like or None, shape (ndof,)
            Initial accelerations. If None, assumed zero.
        qddf : array-like or None, shape (ndof,)
            Final accelerations. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        qdd0 = np.zeros_like(q0) if qdd0 is None else np.asarray(qdd0, dtype=float)
        qddf = np.zeros_like(q0) if qddf is None else np.asarray(qddf, dtype=float)

        A = np.array(
                [[1, t0, t0**2, t0**3, t0**4, t0**5],
                 [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                 [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                 [1, tf, tf**2, tf**3, tf**4, tf**5],
                 [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                 [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
                ])

        b = np.vstack([
            q0,
            qd0,
            qdd0,
            qf,
            qdf,
            qddf
        ])
        self.coeff = np.linalg.solve(A, b)

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof):
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2 + 4 * c[4] * t**3 + 5 * c[5] * t**4
            qdd = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3

            X[i, 0, :] = q
            X[i, 1, :] = qd
            X[i, 2, :] = qdd

        return t, X

class Trapezoidal():
    """
    Trapezoidal velocity profile trajectory generation.
    """

    def __init__(self, ndof=None):
        self.ndof = ndof

    def solve(self, q0, qf, qd0=None, qdf=None, qdd0=None, qddf=None, T=1):
        """
        Compute trapezoidal velocity profile parameters for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        T : float
            Total trajectory duration.
        """
        self.q0 = np.asarray(q0, dtype=float)
        self.qf = np.asarray(qf, dtype=float)

        h = self.qf - self.q0
        # Cruise velocity: midpoint of valid range (h/T, 2h/T)
        self.V = 1.5 * h / T
        # Blend time: 
        # tb = (q0 - qf + V*T) / V = 
        # T - (q0 - qf) / V =
        # T - h / V =
        # T - h / (1.5 * h / T) =
        # T - 2/3 * T =
        # T/3
        self.tb = T / 3.0
        # Acceleration during blend phase
        self.alpha = self.V / self.tb

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))

        for j in range(self.ndof):
            for k, tk in enumerate(t):
                if tk < self.tb:
                    X[j, 0, k] = self.q0[j] + 0.5 * self.alpha[j] * tk**2
                    X[j, 1, k] = self.alpha[j] * tk
                    X[j, 2, k] = self.alpha[j]
                elif tk <= (tf - self.tb):
                    X[j, 0, k] = self.q0[j] + self.V[j] * (tk - self.tb / 2)
                    X[j, 1, k] = self.V[j]
                    X[j, 2, k] = 0
                else:
                    X[j, 0, k] = self.qf[j] - 0.5 * self.alpha[j] * (tf - tk)**2
                    X[j, 1, k] = self.alpha[j] * (tf - tk)
                    X[j, 2, k] = -self.alpha[j]

        return t, X


class MultiAxisTrajectoryGenerator():
    """
    Multi-axis trajectory generator for joint or task space trajectories.

    Supports cubic, quintic polynomial, and trapezoidal velocity profiles.
    """
    
    def __init__(self, method=None,
                 mode="joint",
                 ndof=1,
                 ):
        """
        Initialize the trajectory generator with the given configuration.
        """
        if method is None:
            raise ValueError("Trajectory generation method must be specified.")
        
        self.ndof = ndof
        self.method = method
        
        if mode == "joint":
            self.mode = "Joint Space"
            self.labels = [f'axis{i+1}' for i in range(self.ndof)]
        elif mode == "task":
            self.mode = "Task Space"
            self.labels = ['x', 'y', 'z']  


    def solve(self, q0, qf, qd0=None, qdf=None, qdd0=None, qddf=None, T=1):
        """
        Fit the trajectory and solve for the trajectory parameters.

        Args:
            q0   : start position
            qf   : final position
            qd0  : start velocity (optional)
            qdf  : final velocity (optional)
            qdd0 : start acceleration (optional)
            qddf : final acceleration (optional)
            T    : duration of the trajectory
        """
        self.T = T
        if isinstance(self.method, CubicPolynomial):
            self.method.solve(q0, qf, qd0, qdf, T=T)
        elif isinstance(self.method, QuinticPolynomial):
            self.method.solve(q0, qf, qd0, qdf, qdd0, qddf, T=T)
        elif isinstance(self.method, Trapezoidal):
            self.method.solve(q0, qf, T=T)

    def generate(self, nsteps=100):
        """
        Generate the trajectory at discrete time steps.

        Args:
            nsteps (int): Number of time steps.
        Returns:
            list: List of position, velocity, acceleration for each DOF.
        """
        self.t, self.X = self.method.generate(tf=self.T, nsteps=nsteps)


    def plot(self):
        """
        Plot the position, velocity, and acceleration trajectories.
        """
        self.fig = plt.figure()
        self.sub1 = self.fig.add_subplot(3,1,1)  # Position plot
        self.sub2 = self.fig.add_subplot(3,1,2)  # Velocity plot
        self.sub3 = self.fig.add_subplot(3,1,3)  # Acceleration plot

        self.fig.set_size_inches(8, 10)
        self.fig.suptitle(self.mode + " Trajectory Generator (Point-to-Point)", fontsize=16)

        colors = ['r', 'g', 'b', 'm', 'y']

        for i in range(self.ndof):
            # position plot
            self.sub1.plot(self.t, self.X[i][0], colors[i]+'o-', label=self.labels[i])
            self.sub1.set_ylabel('position', fontsize=15)
            self.sub1.grid(True)
            self.sub1.legend()
        
            # velocity plot
            self.sub2.plot(self.t, self.X[i][1], colors[i]+'o-', label=self.labels[i])
            self.sub2.set_ylabel('velocity', fontsize=15)
            self.sub2.grid(True)
            self.sub2.legend()

            # acceleration plot
            self.sub3.plot(self.t, self.X[i][2], colors[i]+'o-', label=self.labels[i])
            self.sub3.set_ylabel('acceleration', fontsize=15)
            self.sub3.set_xlabel('Time (secs)', fontsize=18)
            self.sub3.grid(True)
            self.sub3.legend()

        plt.show()
        

class MultiSegmentTrajectoryGenerator():
    """
    Multi-segment trajectory generator.

    Output:
        t : (N,)
        X : (ndof, 3, N)
    """
    
    def __init__(self, method=None,
                 mode="joint",
                 ndof=1
                 ):
        """
        Initialize the trajectory generator with the given configuration.
        """
        if method is None:
            raise ValueError("Trajectory generation method must be specified.")
        
        self.ndof = ndof
        self.method = method

        if mode == "joint":
            self.mode = "Joint Space"
            self.labels = [f'axis{i+1}' for i in range(self.ndof)]
        elif mode == "task":
            self.mode = "Task Space"
            self.labels = ['x', 'y', 'z']

    
    def solve(self, waypoints, T):
        """
        Fit the trajectories and solve for the trajectory parameters.

        Args:
            waypoints : (N, ndof)
            T         : segment duration
        """

        wp = np.asarray(waypoints, dtype=float)

        self.waypoints = wp
        self.n_segments = wp.shape[0] - 1
        self.T = T

        # Piecewise polynomial case
        self.segment_models = []
        for i in range(self.n_segments):
            # position constraints at waypoints
            q0 = wp[i]
            qf = wp[i + 1]

            # Velocity continuity at waypoints
            if i == 0: # first segment, start velocity is zero
                qd0 = np.zeros(self.ndof)
            else:
                qd0 = (wp[i] - wp[i - 1]) / self.T

            if i == self.n_segments - 1: # last segment, final velocity is zero
                qdf = np.zeros(self.ndof)
            else:
                qdf = (wp[i + 1] - wp[i]) / self.T

            if isinstance(self.method, QuinticPolynomial):
                qdd0 = np.zeros(self.ndof)
                qddf = np.zeros(self.ndof)

            print(f"Segment {i+1}: q0={q0}, qf={qf}, qd0={qd0}, qdf={qdf}")

            model = type(self.method)(ndof=self.ndof) # creates a new instance of the trajectory gen class

            if isinstance(self.method, CubicPolynomial):
                model.solve(q0, qf, qd0, qdf, T=T)
            elif isinstance(self.method, QuinticPolynomial):
                model.solve(q0, qf, qd0, qdf, qdd0, qddf, T=T)
            elif isinstance(self.method, Trapezoidal):
                model.solve(q0, qf, T=T)


            self.segment_models.append(model)

    
    def generate(self, nsteps_per_segment=100):
        """
        Generate trajectory.

        Args:
            nsteps_per_segment : number of samples per segment

        Returns:
            t : (N,)
            X : (ndof, 3, N)
        """
        segments_X = []

        total_nsteps = (nsteps_per_segment-1) * self.n_segments + 1 # to avoid duplicate points at segment boundaries
        self.t = np.linspace(0, self.T, total_nsteps)

        for i, model in enumerate(self.segment_models):
            _, X_ = model.generate(tf=self.T, nsteps=nsteps_per_segment)

            if i > 0: # Avoid duplicate points at boundaries
                X_ = X_[:, :, 1:]

            segments_X.append(X_)

        # Concatenate all segments
        self.X = np.concatenate(segments_X, axis=2)

    
    def plot(self):
        """
        Plot trajectory (position, velocity, acceleration).

        Args:
            t : (N,) time vector
            X : (ndof, 3, N) trajectory array
        """

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Multi-Segment Trajectory", fontsize=16)

        labels = [f'axis{i+1}' for i in range(self.ndof)]
        colors = ['r', 'g', 'b', 'm', 'y']

        for i in range(self.ndof):
            c = colors[i]+'o-'

            axs[0].plot(self.t, self.X[i, 0, :], c, label=labels[i]) # Position
            axs[1].plot(self.t, self.X[i, 1, :], c, label=labels[i]) # Velocity
            axs[2].plot(self.t, self.X[i, 2, :], c, label=labels[i]) # Acceleration

        axs[0].set_ylabel("Position")
        axs[1].set_ylabel("Velocity")
        axs[2].set_ylabel("Acceleration")
        axs[2].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()




