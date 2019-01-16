from cor_control_benchmarks.control_benchmark import *


class RobotNavigationBenchmark(ControlBenchmark):
    """ Drive a robot through a continuous state space and end up in the right place looking in the right direction."""

    def __init__(self,
                 sampling_time: float = 0.2,
                 max_seconds: float = 20,
                 reward_type: RewardType = RewardType.BINARY,
                 do_not_normalize: bool = False
                 ) -> None:
        """ Create an instance of the robot navigation benchmark.
        :param sampling_time: number of seconds between control decisions and observations.
        :param max_seconds: number of seconds per episode
        :param reward_type: the type of reward function to use.
        :param do_not_normalize: do not normalize the interface with the user: return states in the benchmark specific 
        domain and require actions in the benchmark specific domain. 
        """
        super().__init__(
            state_names=['x location [m]', 'y location [m]', 'heading [rad]'],
            action_names=['forward velocity [m/s]', 'angular velocity [rad/s]'],
            state_shift=np.array([0.5, 0.5, 0.]),
            state_scale=np.array([0.5, 0.5, np.pi]),
            action_shift=np.array([0.1, 0.]),
            action_scale=np.array([0.1, 0.5]),
            initial_states=None,  # initial states sampled uniformly at random from the state space
            sampling_time=sampling_time,
            max_seconds=max_seconds,
            target_state=np.array([0.5, 0.5, 0.]),
            target_action=np.array([0., 0.]),
            state_penalty_weights=np.array([1., 1., 1.]),
            action_penalty_weights=np.array([0., 0.]),
            binary_reward_state_tolerance=np.array([0.01, 0.01, 0.02]),
            binary_reward_action_tolerance=np.array([10., 10.]),
            domain_bound_handling=[DomainBound.IGNORE, DomainBound.IGNORE, DomainBound.WRAP],  # [x, y, phi]
            reward_type=reward_type,
            do_not_normalize=do_not_normalize,
        )

    @property
    def reward(self) -> float:
        """Obtain the reward based on the current state and the action that resulted in the transition to that state.
        Scaled here to give rewards of around 0.5 initially."""
        return super().reward

    @property
    def name(self) -> str:
        """Return an identifier that describes the benchmark for fair comparisons."""
        return f'Robot navigator (v0, ts: {self.sampling_time} s, duration: {self.max_seconds} s. {self.reward_type})'

    def _eom(self, state_action: np.ndarray):
        """Equations of motion for robot navigator.
        :param state_action: concatenated state and action
        :return: derivative of the state-action"""

        x = state_action
        dx = np.zeros_like(x)

        # x_pos = x[0]
        # y_pos = x[1]
        heading = x[2]
        velocity_command_forward = x[3]
        velocity_command_angular = x[4]

        # derivative x pos
        dx[0] = velocity_command_forward * np.cos(heading)

        # derivative y pos
        dx[1] = velocity_command_forward * np.sin(heading)

        # derivative heading
        dx[2] = velocity_command_angular

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        return [-1, -1, -1][state_dimension]
