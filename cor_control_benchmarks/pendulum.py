from cor_control_benchmarks.control_benchmark import *


class PendulumBenchmark(ControlBenchmark):
    """ Swing up an under-actuated pendulum. To solve the benchmark the pendulum will need to be swung to one side to
    build momentum before swinging in the opposite direction and stabilizing in the upright position.

    The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
    DCSC / CoR lab.
    """

    def __init__(self,
                 sampling_time: float = 0.02,
                 max_seconds: float = 5.,
                 reward_type: RewardType = RewardType.QUADRATIC,
                 max_voltage: float = 2.,
                 do_not_normalize: bool = False,
                 state_penalty_weights: np.ndarray = np.array([5., 0.1]),
                 action_penalty_weights: np.ndarray = np.array([1.]),
                 ) -> None:
        """ Create an instance of the pendulum benchmark.
        :param sampling_time: number of seconds between control decisions and observations.
        :param max_seconds: number of seconds per episode
        :param reward_type: the type of reward function to use.
        :param max_voltage: the maximum voltage to the motor, lower values require more swings to reach the top
        :param do_not_normalize: do not normalize the interface with the user: return states in the benchmark specific
        domain and require actions in the benchmark specific domain.
        """

        super().__init__(
            state_names=['pendulum angle [rad]', 'angular velocity [rad/s]'],
            action_names=['motor voltage [V]'],
            state_shift=np.array([0., 0.]),
            state_scale=np.array([np.pi, 30]),  # states in  [-pi, pi], [-30, 30]
            action_shift=np.array([0.]),
            action_scale=np.array([max_voltage]),  # actions in [-max_voltage, max_voltage]
            initial_states=[
                np.array([0., 0.]),
            ],
            sampling_time=sampling_time,
            max_seconds=max_seconds,
            target_state=np.array([np.pi, 0.]),
            target_action=np.array([0.]),
            state_penalty_weights=state_penalty_weights,
            action_penalty_weights=action_penalty_weights,
            binary_reward_state_tolerance=np.array([0.05, 0.1]),
            binary_reward_action_tolerance=np.array([0.1]),
            domain_bound_handling=[DomainBound.WRAP, DomainBound.IGNORE],  # 'Pendulum angle, angular velocity'
            reward_type=reward_type,
            do_not_normalize=do_not_normalize,
        )

    @property
    def reward(self) -> float:
        """Obtain the reward based on the current state and the action that resulted in the transition to that state.
        Scaled here to give rewards of around 0.5 initially."""
        return super().reward / 100

    @property
    def name(self) -> str:
        """Return an identifier that describes the benchmark for fair comparisons."""
        return f'pendulum swingup (v0, ' \
            f'max voltage: {self.action_scale[0]}, st: {self.sampling_time} s, duration: {self.max_seconds} s,  ' \
            f'{self.reward_type})'

    def _eom(self, state_action: np.ndarray):
        """Equations of motion for DCSC/CoR inverted pendulum setup.
        :param state_action: concatenated state and action
        :return: derivative of the state-action"""

        x = state_action
        dx = np.zeros_like(x)

        angle = x[0]
        angular_velocity = x[1]
        motor_torque = x[2]

        inertia = 1.91e-4  # Pendulum inertia
        mass = 5.5e-2  # Pendulum mass
        gravity_constant = 9.81  # Gravity constant
        length = 4.2e-2  # Pendulum length
        damping = 3e-6  # Viscous damping
        torque_constant = 5.36e-2  # Torque constant
        rotor_resistance = 9.5  # Rotor resistance

        # derivative of the angle
        dx[0] = angular_velocity

        # derivative of the angular velocity
        dx[1] = (
                        - mass * gravity_constant * length * np.sin(angle)
                        - (damping + torque_constant ** 2 / rotor_resistance) * angular_velocity
                        + torque_constant / rotor_resistance * motor_torque
                ) / inertia

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        return [1, -1, -1][state_dimension]
