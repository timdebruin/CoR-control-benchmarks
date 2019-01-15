from cor_control_benchmarks.control_benchmark import *


class SegwayBenchmark(ControlBenchmark):
    """ Model of a Segway. The segway starts out under a significant angle and the control challenge is to right it
    quickly and stabilize. """

    def __init__(self,
                 sampling_time: float = 0.01,
                 max_seconds: float = 2.5,
                 reward_type: RewardType = RewardType.ABSOLUTE):
        """ Create an instance of the segway benchmark.
        :param sampling_time: number of seconds between control decisions and observations.
        :param max_seconds: number of seconds per episode
        :param reward_type: the type of reward function to use. """
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
        """Equations of motion of a segway model.
        :param state_action: concatenated state and action
        :return: derivative of the state-action"""
        x = state_action
        body_angle = x[0]
        body_angular_velocity = x[1]
        wheel_velocity = x[2]
        motor_torque = x[3]

        # 'Body angle', 'Body velocity', 'Wheel velocity', 'action'
        dx = np.zeros_like(x)

        gravity_constant = 9.8  # gravity constant
        half_length_link = 0.25  # half of the length of the link
        radius_wheel = 0.1  # radius of the wheel
        mass_wheel = 0.1  # mass of the wheel
        mass_link = 8  # mass of the link
        inertia_wheel = 0.0005  # inertia of the wheel
        inertia_link = 0.227267  # inertia of the link
        friction_axle = 0.01  # axle friction
        friction_rolling = 0.1  # rolling friction

        # derivative of body angle
        dx[0] = body_angular_velocity

        # derivative of body angular velocity
        dx[1] = (
                - (
                        (inertia_wheel + (mass_wheel + mass_link) * radius_wheel ** 2) *
                        (
                                - motor_torque
                                + gravity_constant * half_length_link * mass_link * np.sin(body_angle)
                                - friction_axle * body_angular_velocity + friction_axle * wheel_velocity
                        )
                        - half_length_link * mass_link * radius_wheel * np.cos(body_angle) *
                        (
                                motor_torque
                                + friction_axle * body_angular_velocity
                                + half_length_link * mass_link * radius_wheel * np.sin(body_angle) *
                                    body_angular_velocity ** 2.
                                - (friction_rolling + friction_axle) * wheel_velocity
                        )
                ) / (
                        - (inertia_link + half_length_link ** 2 * mass_link) *
                        (
                            inertia_wheel + (mass_wheel + mass_link) * radius_wheel ** 2
                        )
                        + half_length_link ** 2 * mass_link ** 2 * radius_wheel ** 2 * np.cos(body_angle) ** 2
                )
        )

        # derivative of wheel velocity
        dx[2] = (
                (
                        -gravity_constant * half_length_link ** 2 * mass_link ** 2 * radius_wheel * np.cos(body_angle) *
                            np.sin(body_angle) + half_length_link * mass_link *
                            (inertia_link + half_length_link ** 2 * mass_link) * radius_wheel * np.sin(body_angle) *
                            body_angular_velocity ** 2
                        + half_length_link * mass_link * radius_wheel * np.cos(body_angle) *
                            (motor_torque + friction_axle * body_angular_velocity - friction_axle * wheel_velocity)
                        + (inertia_link + half_length_link ** 2 * mass_link) *
                            (motor_torque + friction_axle * body_angular_velocity
                             - (friction_rolling + friction_axle) * wheel_velocity
                            )
                ) /
                (
                        (inertia_link + half_length_link ** 2 * mass_link) *
                            (
                                    inertia_wheel
                                    + (mass_wheel + mass_link) * radius_wheel ** 2
                            )
                            - half_length_link ** 2 * mass_link ** 2 * radius_wheel ** 2 *
                                np.cos(body_angle) ** 2
                )
        )

        return dx

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        return [1, -1, -1][state_dimension]
