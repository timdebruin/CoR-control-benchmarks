from cor_control_benchmarks.control_benchmark import *


class MagmanBenchmark(ControlBenchmark):
    """ Position a magnetic ball by controlling the current through several electromagnets positioned under a
    1-dimensional track that the ball rolls on.

    The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
    DCSC / CoR lab.
    """

    def __init__(self,
                 sampling_time: float = 0.02,
                 max_seconds: float = 3.,
                 reward_type: RewardType = RewardType.QUADRATIC,
                 magnets: int = 4,
                 do_not_normalize: bool = False,
                 ) -> None:
        """ Create an instance of the pendulum benchmark.
        :param sampling_time: number of seconds between control decisions and observations.
        :param max_seconds: number of seconds per episode
        :param reward_type: the type of reward function to use.
        :param magnets: the number of magnets (action dimensionality). [1 - 4] note that for one magnet the problem
        changes significantly (becomes harder) as a ballistic trajectory needs to be learned
        :param do_not_normalize: do not normalize the interface with the user: return states in the benchmark specific
        domain and require actions in the benchmark specific domain."""
        super().__init__(
            state_names=['ball position [m]', 'ball velocity [m/s]'],
            action_names=[f'current magnet {i+1}' for i in range(magnets)],
            state_shift=np.array([0.035, 0.]),
            state_scale=np.array([0.07, 0.4]),
            action_shift=np.array([0.3 for _ in range(magnets)]),
            action_scale=np.array([0.3 for _ in range(magnets)]),
            initial_states=[
                np.array([0., 0.]),
            ],
            sampling_time=sampling_time,
            max_seconds=max_seconds,
            target_state=np.array([0.035, 0.]),
            target_action=np.array([0. for _ in range(magnets)]),
            state_penalty_weights=np.array([1., 0.]),
            action_penalty_weights=np.array([0. for _ in range(magnets)]),
            binary_reward_state_tolerance=np.array([0.01, 0.01]),
            binary_reward_action_tolerance=np.array([1. for _ in range(magnets)]),
            domain_bound_handling=[DomainBound.STOP, DomainBound.IGNORE],
            #                       Ball position, ball velocity
            reward_type=reward_type,
            do_not_normalize=do_not_normalize,
        )
        self.magnets = magnets

    @property
    def name(self) -> str:
        """Return an identifier that describes the benchmark for fair comparisons."""
        return f'Magman (v0, magnets: {self.magnets}, st: {self.sampling_time}, duration: {self.max_seconds} s, ' \
            f'{self.reward_type})'

    def _eom(self, state_action: np.ndarray):
        """Equations of motion for the DCSC/CoR magman setup.
        :param state_action: concatenated state and action
        :return: derivative of the state-action"""
        x = state_action
        dx = np.zeros_like(x)

        # Equations of motion for the DCSC/CoR magnetic manipulation setup
        alpha = 5.52e-10  # magnetic force function parameter
        beta = 1.75e-4  # magnetic force function parameter
        friction = 0.0161  # viscous friction coefficient
        mass = 0.032  # ball mass

        ball_position = x[0]
        ball_velocity = x[1]

        # position derivative
        dx[0] = ball_velocity

        magnetic_force = 0
        for magnet_index in range(self.magnets):
            squared_current = x[2 + magnet_index]
            magnet_position = (magnet_index + 1) * 0.025  # magnets are 25 mm apart, starting fom pos x[0]=0.025
            magnetic_force += (
                    squared_current * (-alpha * (ball_position - magnet_position)) /
                    ((((ball_position - magnet_position) ** 2) + beta) ** 3)
            )

        # velocity derivative
        dx[1] = (
                        - friction * ball_velocity
                        + magnetic_force
                ) / mass

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        return ([1, -1] + [-1 for _ in range(self.magnets)])[state_dimension]
