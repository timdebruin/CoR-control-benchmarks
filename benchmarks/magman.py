from benchmarks.control_benchmark import *


class MagmanBenchmark(ControlBenchmark):

    def __init__(self,
                 sampling_time: float = 0.02,
                 max_seconds: float = 2.5,
                 reward_type: RewardType = RewardType.QUADRATIC,
                 magnets: int = 4):
        super().__init__(
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
            action_penalty_weights=np.array([1. for _ in range(magnets)]),
            domain_bound_handling=[DomainBound.STOP, DomainBound.IGNORE],
            # Ball position, ball velocity
            reward_type=reward_type,
        )
        self.magnets = magnets

    @property
    def name(self) -> str:
        return f'magman-magnets_{self.magnets}-ts_{self.sampling_time}-ms_{self.max_seconds}-rt_{self.reward_type}'

    def _eom(self, state_action: np.ndarray):
        x = state_action

        # Equations of motion for the DCSC/CoR magnetic manipulation setup
        alpha = 5.52e-10  # magnetic force function parameter
        beta = 1.75e-4  # magnetic force function parameter
        friction = 0.0161  # viscous friction coefficient
        mass = 0.032  # ball mass

        dx = np.zeros_like(x)

        ball_position = x[0]
        ball_velocity = x[1]

        # position derivative
        dx[0] = ball_velocity

        magnetic_force = 0
        for magnet_index in range(self.magnets):
            squared_current = x[2 + magnet_index]
            magnet_position = magnet_index * 0.025  # magnets are 25 mm apart, starting fom pos x[0]=0
            magnetic_force += squared_current * (-alpha * (ball_position - magnet_position) /
                                                 ((ball_position - magnet_position) ** 2) + beta) ** 3

        # velocity derivative
        dx[1] = (
                    - friction * ball_velocity
                    + magnetic_force
                ) / mass 

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        return ([1, -1] + [-1 for _ in range(self.magnets)])[state_dimension]


if __name__ == '__main__':
    m = MagmanBenchmark(magnets=3)
    assert np.allclose(m.denormalize_action(np.array([-1., 0., 1.])), np.array([0., 0.3, 0.6]))
    assert np.allclose(m.denormalize_state(np.array([-1., 0.])), np.array([-0.035, 0.0]))
    assert np.allclose(m.normalize_state(np.array([0.105, 0.4])), np.array([1., 1.]))
