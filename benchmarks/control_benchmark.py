import random
from typing import Union, Any, Tuple, List
from enum import Enum

# import gym.spaces
import numpy as np

from result_logging.trajectory_logging import TrajectoryLogger, DummyLogger


class RewardType(Enum):
    """The type of penalization used on the difference between the target and real state and control action."""
    ABSOLUTE = 1  # Wx * |x - xr| + Wu * |u - ur|
    QUADRATIC = 2  # Wx * |x - xr|^2 + Wu * |u - ur|^2
    BINARY = 3  # 1 if euclidean distance of (x,xr) and of (u, ur) < tolerance, else 0


class DomainBound(Enum):
    """What to do when leaving the normalized benchmark domain. Set per state-dimension. """
    WRAP = 1  # transport to the other end of the normalized state component domain
    TERMINATE = 2  # terminate the episode when leaving the state component domain
    STOP = 3  # clip the state component, set derivative of the state component to zero when clipping (if the
    # derivative is in the state)
    IGNORE = 4  # allow the state component to diverge from the normalized domain


class ControlBenchmark(object):
    """The control benchmark base class. Do not use directly, use a subclass.

    The objective of these benchmarks is make the state of a dynamical system converge to a target state.
    """

    def __init__(self,
                 state_shift: np.ndarray, state_scale: np.ndarray,
                 action_shift: np.ndarray, action_scale: np.ndarray,
                 initial_states: List[np.ndarray],
                 sampling_time: float,
                 max_seconds: float,
                 target_state: np.ndarray,
                 target_action: np.ndarray,
                 state_penalty_weights: np.ndarray,
                 action_penalty_weights: np.ndarray,
                 domain_bound_handling: List[DomainBound],
                 reward_type: RewardType) -> None:
        """Initialization function of the control benchmark base class.
        Should be called from the init functions of the actual benchmarks.

        :param state_shift:
        :param state_scale:
        :param action_shift:
        :param action_scale:
        :param initial_states:
        :param sampling_time:
        :param max_seconds:
        :param target_state:
        :param target_action:
        :param state_penalty_weights:
        :param action_penalty_weights:
        :param domain_bound_handling:
        :param reward_type:
        """

        assert len(domain_bound_handling) == len(state_scale) == len(state_shift) == len(initial_states[0]) \
            == len(state_penalty_weights), \
            'all state related quantities should have the same dimensions'
        assert len(action_shift) == len(action_scale) == len(action_penalty_weights) == len(target_action), \
            'all action related quantities should have the same dimensions'

        self.target_state_action = np.concatenate((target_state, target_action))
        self.state_action_penalty_weights = np.concatenate((state_penalty_weights, action_penalty_weights))
        assert self.target_state_action.shape == self.state_action_penalty_weights.shape
        self.reward_type = reward_type
        self.sampling_time = sampling_time  # in seconds
        self.max_seconds = max_seconds
        self.initial_states = initial_states  # in benchmark coordinates (not normalized)
        self.action_scale = action_scale
        self.action_shift = action_shift
        self.state_scale = state_scale
        self.state_shift = state_shift
        self.domain_bound_handling = domain_bound_handling

        self.step_counter = 0
        self._state: np.ndarray = None
        self._u: np.ndarray = None

        self.logging: TrajectoryLogger = DummyLogger()  # Result logging can be attached later

        # Variables only used for compatibility with functions that expect an OpenAI gym environment
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.action_shift),), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.state_shift),), dtype=np.float32)
        # self.metadata = None
        # self.reward_range = None
        # self.spec = None
        # self.unwrapped = self

    def reset(self) -> np.ndarray:
        """Reset the environment to one of the initial states"""
        self._state = random.choice(self.initial_states)
        self.step_counter = 0
        self._reset_log()
        return self.normalized_state

    def _reset_log(self) -> None:
        self.logging.reset_log()

    def reset_to_true_state(self, true_state: np.ndarray):
        """Reset the environment to a specific state in the benchmark domain (non normalized)"""
        self._state = true_state
        self.step_counter = 0
        return self.normalized_state

    @property
    def normalized_state(self) -> np.ndarray:
        """The normalized state of the environment"""
        return self.normalize_state(self.true_state)

    @property
    def true_state(self) -> np.ndarray:
        """The non normalized state of the environment"""
        return np.array(self._state)

    def _f(self):
        """Calculate the state at the next time step based on the equations of motion.
        Uses two steps of the Runge–Kutta method (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)"
        """
        y = np.concatenate((self._state, self._u))
        k1 = self._eom(y)
        k2 = self._eom(y + 0.25 * self.sampling_time * k1)
        k3 = self._eom(y + 0.25 * self.sampling_time * k2)
        k4 = self._eom(y + 0.5 * self.sampling_time * k3)
        y = y + (self.sampling_time / 12) * (k1 + 2 * k2 + 2 * k3 + k4)
        k1 = self._eom(y)
        k2 = self._eom(y + 0.25 * self.sampling_time * k1)
        k3 = self._eom(y + 0.25 * self.sampling_time * k2)
        k4 = self._eom(y + 0.5 * self.sampling_time * k3)
        y = y + (self.sampling_time / 12) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = y[:len(self._state)]

    def _eom(self, state_action: np.ndarray) -> np.ndarray:
        """Calculates the benchmark specific equations of motion for the state-action vector"""
        raise NotImplementedError('Do not use this class directly, use a subclass')

    def step(self, action: Union[np.ndarray]) -> Tuple[np.ndarray, float, bool, Any]:
        """ Take a step from the current state using the specified action. Action is assumed to be in [-1,1]
        :param action: The action to take in the current state for sample_time seconds
        :return: A tuple with (next_normalized_state, reward, terminal, additional_info)
        """
        action = np.array(action)
        assert -1 <= action.all() <= 1, 'Actions should be normalized between -1 and 1'
        self._u = self.denormalize_action(action)
        self._step_log_pre()
        self._f()
        self.step_counter += 1
        terminal = self._state_bounds_check() or self.max_steps_passed
        reward = self.reward

        self._step_log_post(reward, terminal)
        return self.normalized_state, reward, terminal, None

    def _step_log_pre(self) -> None:
        """ Logs the state-action (before dynamics function is called)"""
        self.logging.step_log_pre(self.true_state, self._u)

    def _step_log_post(self, reward, terminal) -> None:
        """ Logs the reward and logs sequence based on terminal and reward sum (after dynamics function is called)"""
        self.logging.step_log_post(reward, terminal)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """ Normalize the state from the benchmark domain to the domain of [-1, 1]^N
        :param state: The non normalized state
        :return: The normalized state
        """
        return self._norm_or_de_norm(state, self.state_shift, self.state_scale, de_norm=False)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """ Obtain the action in the de-normalized (benchmark) form
        :param action: action in normalized form ( in [-1, 1]^N )
        :return: the action in the de-normalized form
        """
        return self._norm_or_de_norm(action, self.action_shift, self.action_scale, de_norm=True)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """ Obtain the state in the de-normalized (benchmark) coordinates
        :param : state in normalized form ( in [-1, 1]^N )
        :return: the action in the de-normalized form
        """
        return self._norm_or_de_norm(state, self.state_shift, self.state_scale, de_norm=True)

    @staticmethod
    def _norm_or_de_norm(vector: np.ndarray, shift: np.ndarray, scale: np.ndarray, de_norm: bool):
        return vector * scale + shift if de_norm else (vector - shift) / scale

    @property
    def reward(self) -> float:
        """Obtain the reward based on the current state and the action that resulted in the transition to that state."""
        difference = np.abs(np.concatenate((self._state, self._u)) - self.target_state_action)
        if self.reward_type == RewardType.ABSOLUTE:
            return float(-1*np.sum(difference * self.state_action_penalty_weights))
        elif self.reward_type == RewardType.QUADRATIC:
            return float(-1*np.sum(difference ** 2 * self.state_action_penalty_weights))
        else:
            raise NotImplementedError(f'No implementation for {self.reward_type}')

    def _state_bounds_check(self) -> bool:
        """Check whether the state has exited the normalized domain,
        correct the state based on the strategy per state component given in domain_bound_handling,
        return whether the domain violation means that the episode should be terminated."""
        for state_index, (normalized_state_component, handling) in enumerate(
                zip(self.normalized_state, self.domain_bound_handling)):
            if normalized_state_component < -1 or normalized_state_component > 1:
                if handling == DomainBound.IGNORE:
                    pass
                elif handling == DomainBound.STOP:
                    corrected_state = self.normalized_state
                    corrected_state[state_index] = np.clip(normalized_state_component, -1, 1)
                    try:
                        corrected_state[self._derivative_dimension(state_index)] = 0.
                    except IndexError:  # derivative is not in the state component
                        pass
                    self._state = self.denormalize_state(corrected_state)
                    return self._state_bounds_check()
                elif handling == DomainBound.WRAP:
                    corrected_state = self.normalized_state
                    # assuming violations are small, but if they are not there is some issue with the
                    # state bounds, sampling frequency or action magnitude
                    corrected_state[state_index] = normalized_state_component - 1 * np.sign(normalized_state_component)
                    self._state = self.denormalize_state(corrected_state)
                    return self._state_bounds_check()
                elif handling == DomainBound.TERMINATE:
                    return True
        return False

    def _derivative_dimension(self, state_dimension: int) -> int:
        """ Return the index in the state vector of the derivative of the state_dimension index
        Return -1 if the derivative of the given state component is not in the state vector
        :param state_dimension: the index in the state vector that you want the derivative of
        :return: the index of the state vector dimension that contains the derivative
        """
        raise NotImplementedError('Should be implemented in the derived class '
                                  'as it is benchmark dependent how the state is defined')

    @property
    def name(self) -> str:
        raise NotImplementedError('Should be implemented in the derived class '
                                  'as it is benchmark dependent')

    @property
    def max_steps_passed(self):
        return self.step_counter * self.sampling_time > self.max_seconds
