from benchmarks.control_benchmark import *


class SegwayBenchmark(ControlBenchmark):

    def __init__(self,
                 sampling_time: float = 0.01,
                 max_seconds: float = 2.5,
                 reward_type: RewardType = RewardType.ABSOLUTE):
        super().__init__(
            state_shift=np.array([0., 0., 0.]),
            state_scale=np.array([3 * np.pi / 5, 12, 180]),
            action_shift=np.array([0.]),
            action_scale=np.array([20.]),
            initial_states=[
                np.array([-2 * np.pi / 5, 0., 0.]),
                np.array([2 * np.pi / 5, 0., 0.]),
            ],
            sampling_time=sampling_time,
            max_seconds=max_seconds,
            target_state=np.array([0., 0., 0.]),
            target_action=np.array([0.]),
            state_penalty_weights=np.array([1., 0., 0.]),
            action_penalty_weights=np.array([0.]),
            domain_bound_handling=[DomainBound.STOP, DomainBound.IGNORE, DomainBound.IGNORE],
            # 'Body angle', 'Body velocity', 'Wheel velocity'
            reward_type=reward_type,
        )

    @property
    def name(self) -> str:
        return f'segway-ts_{self.sampling_time}-ms_{self.max_seconds}-rt_{self.reward_type}'

    def _eom(self, state_action: np.ndarray):
        x = state_action

        g = 9.8
        length = 0.25
        r = 0.1
        m1 = 0.1
        m2 = 8
        i1 = 0.0005
        i2 = 0.227267
        f_delta = 0.01
        f_beta = 0.1

        dx = np.array([0., 0., 0., 0.])
        # 'Body angle', 'Body velocity', 'Wheel velocity', 'action'

        # Body angle
        dx[0] = x[1]

        # Body velocity
        dx[1] = -((i1 + (m1 + m2) * r ** 2) *
                  (-x[3] + g * length * m2 * np.sin(x[0]) - f_delta * x[1] + f_delta * x[2]) -
                  length * m2 * r * np.cos(x[0]) * (x[3] + f_delta * x[1] + length * m2 * r * np.sin(x[0]) * x[1] ** 2.
                                                    - (f_beta + f_delta) * x[2])) / (
                        -(i2 + length ** 2 * m2) * (i1 + (m1 + m2) * r ** 2) +
                        length ** 2 * m2 ** 2 * r ** 2 * np.cos(x[0]) ** 2)

        # Wheel velocity
        dx[2] = (-g * length ** 2 * m2 ** 2 * r * np.cos(x[0]) * np.sin(x[0]) + length * m2 *
                 (i2 + length ** 2 * m2) * r * np.sin(x[0]) * x[1] ** 2 + length * m2 * r * np.cos(x[0]) *
                 (x[3] + f_delta * x[1] - f_delta * x[2])
                 + (i2 + length ** 2 * m2) * (x[3] + f_delta * x[1] - (f_beta + f_delta) * x[2])) / \
                ((i2 + length ** 2 * m2) * (i1 + (m1 + m2) * r ** 2) - length ** 2 * m2 ** 2 * r ** 2 * np.cos(
                    x[0]) ** 2)

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        return [1, -1, -1][state_dimension]
