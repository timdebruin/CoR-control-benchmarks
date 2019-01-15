from benchmarks.control_benchmark import *


class PendulumBenchmark(ControlBenchmark):

    def __init__(self,
                 sampling_time: float = 0.02,
                 max_seconds: float = 4.0,
                 reward_type: RewardType = RewardType.QUADRATIC,
                 ):
        super().__init__(
            state_shift=np.array([0., 0.]),
            state_scale=np.array([np.pi, 30]),  # states in  [-pi, pi], [-30, 30]
            action_shift=np.array([0.]),
            action_scale=np.array([3.]),  # actions in [-3V, 3V]
            initial_states=[
                np.array([0., 0.]),
            ],
            sampling_time=sampling_time,
            max_seconds=max_seconds,
            target_state=np.array([np.pi, 0.]),
            target_action=np.array([0.]),
            state_penalty_weights=np.array([5., 0.1]),
            action_penalty_weights=np.array([1.]),
            domain_bound_handling=[DomainBound.WRAP, DomainBound.IGNORE],  # 'Pendulum angle, angular velocity'
            reward_type=reward_type,
        )

    @property
    def reward(self) -> float:
        return super().reward / 100

    @property
    def name(self) -> str:
        return f'pendulum_swingup_v0-ts_{self.sampling_time}-ms_{self.max_seconds}-rt_{self.reward_type}'

    def _eom(self, state_action: np.ndarray):
        """Equations of motion for DCSC/CoR inverted pendulum setup
        :param state_action: concatenated state and action
        :return: derivative of the state-action
        """
        x = state_action
        angle = x[0]
        angular_velocity = x[1]

        inertia = 1.91e-4  # Pendulum inertia
        mass = 5.5e-2  # Pendulum mass
        gravity_constant = 9.81  # Gravity constant
        length = 4.2e-2  # Pendulum length
        damping = 3e-6  # Viscous damping
        torque_constant = 5.36e-2  # Torque constant
        rotor_resistance = 9.5  # Rotor resistance

        dx = np.zeros((3,))

        # derivative of the angle
        dx[0] = angular_velocity

        # derivative of the angular velocity
        dx[1] = (
                        - mass * gravity_constant * length * np.sin(angle)
                        - (damping + torque_constant ** 2 / rotor_resistance) * angular_velocity
                        + torque_constant / rotor_resistance * angular_velocity
                ) / inertia

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        return [1, -1, -1][state_dimension]


if __name__ == '__main__':
    m = PendulumBenchmark()
    assert np.allclose(m.denormalize_action(np.array([-1.])), np.array([-3.]))
    assert np.allclose(m.denormalize_state(np.array([-1., 0.])), np.array([-np.pi, 0.0]))
    assert np.allclose(m.normalize_state(np.array([np.pi, 30])), np.array([1., 1.]))
