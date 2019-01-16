import random
from typing import Union, Any, Tuple, List, Optional, Callable
from enum import Enum

# import gym.spaces
import numpy as np

from cor_control_benchmarks.result_logging.trajectory_logging import TrajectoryLogger


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

    The objective of these cor_control_benchmarks is make the state of a dynamical system converge to a target state.
    """

    def __init__(self,
                 state_names: List[str], action_names: List[str],
                 state_shift: np.ndarray, state_scale: np.ndarray,
                 action_shift: np.ndarray, action_scale: np.ndarray,
                 initial_states: Optional[Union[List[np.ndarray], Callable]],
                 sampling_time: float,
                 max_seconds: float,
                 target_state: np.ndarray,
                 target_action: np.ndarray,
                 state_penalty_weights: np.ndarray,
                 action_penalty_weights: np.ndarray,
                 binary_reward_state_tolerance: np.ndarray,
                 binary_reward_action_tolerance: np.ndarray,
                 domain_bound_handling: List[DomainBound],
                 reward_type: RewardType,
                 do_not_normalize: bool
                 ) -> None:
        """Initialization function of the control benchmark base class.
        Should be called from the init functions of the actual cor_control_benchmarks.

        :param state_shift: shift to use to go between normalized and benchmark states:
         normalized = (benchmark - shift) / scale
        :param state_scale: scaling to use to go between normalized and benchmark states:
         normalized = (benchmark - shift) / scale
        :param action_shift: shift to use to go between normalized and benchmark actions:
         normalized = (benchmark - shift) / scale
        :param action_scale: scale to use to go between normalized and benchmark actions:
         normalized = (benchmark - shift) / scale
        :param initial_states: either None (initial state will be sampled uniformly at random from the state space),
        a list of states that the benchmark can be set to (uniformly at random) when reset,
        or a callable that returns a state.
        :param sampling_time: seconds between subsequent control decisions and observations
        :param max_seconds: number of seconds before an episode terminates
        :param target_state: the desired state the system should be controlled towards
        :param target_action: the desired action
        :param state_penalty_weights: penalization scaling per state dimension
        :param action_penalty_weights: penalization scaling per action dimension
        :param binary_reward_state_tolerance: tolerance per state component when using binary rewards
        :param binary_reward_action_tolerance: tolerance per action component when using binary rewards
        :param domain_bound_handling: how to handle violations of the normalized state domain per state component
        :param reward_type: type of reward function to use
        :param do_not_normalize: do not normalize the interface with the user: return states in the benchmark specific
        domain and require actions in the benchmark specific domain.
        """
        self.state_names = state_names
        self.action_names = action_names
        self.do_not_normalize = do_not_normalize
        assert len(domain_bound_handling) == len(state_scale) == len(state_shift) \
               == len(state_penalty_weights), \
            'all state related quantities should have the same dimensions'
        assert len(action_shift) == len(action_scale) == len(action_penalty_weights) == len(target_action), \
            'all action related quantities should have the same dimensions'

        self.target_state = target_state
        self.target_action = target_action
        self.target_state_action = np.concatenate((target_state, target_action))
        self.state_action_penalty_weights = np.concatenate((state_penalty_weights, action_penalty_weights))
        assert self.target_state_action.shape == self.state_action_penalty_weights.shape
        self.reward_type = reward_type
        self.binary_reward_state_action_tolerance = np.concatenate((binary_reward_state_tolerance,
                                                                    binary_reward_action_tolerance))
        self.sampling_time = sampling_time  # in seconds
        self.max_seconds = max_seconds
        self.initial_states = initial_states  # in benchmark coordinates (not normalized)
        self.action_scale = action_scale
        self.action_shift = action_shift
        self.state_scale = state_scale
        self.state_shift = state_shift
        self.domain_bound_handling = domain_bound_handling

        self.step_counter = 0  # number of discrete time steps elapsed since the start of the current episode
        self._state: np.ndarray = None
        self._u: np.ndarray = None

        self.loggers: List[TrajectoryLogger] = []  # Result loggers can be attached later

        # Variables only used for compatibility with functions that expect an OpenAI gym environment
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.action_shift),), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.state_shift),), dtype=np.float32)
        # self.metadata = None
        # self.reward_range = None
        # self.spec = None
        # self.unwrapped = self

    def reset(self) -> np.ndarray:
        """Reset the environment to one of the initial states.
        :return the initial state after reset"""
        if self.initial_states is None:
            #  randomly sample a state from the state space
            self._state = self.denormalize_state(np.random.uniform(-1., 1., size=self.state_scale.size))
        elif isinstance(self.initial_states, list):
            self._state = random.choice(self.initial_states)
        else:  # only other allowed alternative is a function that returns an initial state
            assert callable(self.initial_states)
            self._state = self.initial_states()
        self.step_counter = 0
        self._reset_log()
        return self.state

    def reset_to_specific_state(self, state: np.ndarray) -> np.ndarray:
        """ Reset the environment to a specific state. If the environment is used with normalized states and actions (
        the default) the given state should be in normalized coordinates. If the environment is used without
        normalization (by passing do_not_normalize=True to the benchmark during initialization) the given
        state should not be normalized.

        :param state: state to reset to
        :return: the state (normalized if the benchmark is used with normalization, and benchmark
        specific otherwise)."""

        self._state = state if self.do_not_normalize else self.denormalize_state(state)

        self.step_counter = 0
        return self.state

    def step(self, action: Union[np.ndarray]) -> Tuple[np.ndarray, float, bool, Any]:
        """ Take a step from the current state using the specified action. Action is assumed to be in [-1,1] unless
        the environment was created with do_not_normalize = True.
        :param action: The action to take in the current state for sample_time seconds
        :return: A tuple with (next_state, reward, terminal, additional_info)"""
        if self.do_not_normalize:
            action = self.normalize_action(np.array(action))
            assert -1 <= action.all() <= 1, 'Action was out of the allowed range'
        else:
            action = np.array(action)
            assert -1 <= action.all() <= 1, 'Actions should be normalized between -1 and 1'
        self._u = self.denormalize_action(action)

        self._step_log_pre()

        self._f()
        self.step_counter += 1
        terminal = self._state_bounds_check() or self.max_steps_passed
        reward = self.reward

        self._step_log_post(reward, terminal)

        return self.state, reward, terminal, None

    @property
    def name(self) -> str:
        """Return an identifier that describes the benchmark for fair comparisons."""
        raise NotImplementedError('Should be implemented in the derived class '
                                  'as it is benchmark dependent')

    @property
    def action_shape(self):
        return self.action_scale.shape

    @property
    def state_shape(self):
        return self.state_scale.shape

    @property
    def not_normalized_state_domain(self):
        return {
            'min': self.denormalize_state(-1 * np.ones(self.state_shape)),
            'max': self.denormalize_state(np.ones(self.state_shape)),
        }

    @property
    def not_normalized_action_domain(self):
        return {
            'min': self.denormalize_action(-1 * np.ones(self.action_shape)),
            'max': self.denormalize_action(np.ones(self.action_shape)),
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """ Normalize the state from the benchmark specific form to the domain of [-1, 1]^N.
        :param state: The non normalized state
        :return: The normalized state """
        return self._norm_or_de_norm(state, self.state_shift, self.state_scale, de_norm=False)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """ Normalize the action from the benchmark specific form to the domain of [-1, 1]^N.
        :param action: The non normalized state
        :return: The normalized action """
        return self._norm_or_de_norm(action, self.action_shift, self.action_scale, de_norm=False)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """ Obtain the action in the de-normalized (benchmark specific) form.
        :param action: action in normalized form ( in [-1, 1]^N )
        :return: the action in the de-normalized form"""
        return self._norm_or_de_norm(action, self.action_shift, self.action_scale, de_norm=True)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """ Obtain the state in the de-normalized (benchmark specific) form.
        :param : state in normalized form ( in [-1, 1]^N )
        :return: the action in the de-normalized form"""
        return self._norm_or_de_norm(state, self.state_shift, self.state_scale, de_norm=True)

    @property
    def state(self) -> np.ndarray:
        """Return either the normalized state or the true state, depending on whether normalization is used"""
        return self.true_state if self.do_not_normalize else self.normalized_state

    @property
    def normalized_state(self) -> np.ndarray:
        """The normalized state of the environment."""
        return self.normalize_state(self.true_state)

    @property
    def true_state(self) -> np.ndarray:
        """The non normalized state of the environment."""
        return np.array(self._state)

    @property
    def reward(self) -> float:
        """Obtain the reward based on the current state and the action that resulted in the transition to that state."""
        difference = np.abs(np.concatenate((self._state, self._u)) - self.target_state_action)
        if self.reward_type == RewardType.ABSOLUTE:
            return float(-1 * np.sum(difference * self.state_action_penalty_weights))
        elif self.reward_type == RewardType.QUADRATIC:
            return float(-1 * np.sum(difference ** 2 * self.state_action_penalty_weights))
        elif self.reward_type == RewardType.BINARY:
            return float(np.all(difference < self.binary_reward_state_action_tolerance))
        else:
            raise NotImplementedError(f'No implementation for {self.reward_type}')

    @property
    def max_steps_passed(self):
        """Returns True if the maximum episode length of the benchmark has expired."""
        return self.step_counter * self.sampling_time > self.max_seconds

    def _f(self):
        """Calculate the state at the next time step based on the equations of motion.
        Uses two steps of the Runge–Kutta method (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)."""
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
        """Calculates the benchmark specific equations of motion for the state-action vector."""
        raise NotImplementedError('Do not use this class directly, use a subclass')

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
        """ Return the index in the state vector of the derivative of the state_dimension index,
        return -1 if the derivative of the given state component is not in the state vector.
        :param state_dimension: the index in the state of the component that the derivative should be of
        :return: the index of the state vector component that contains the derivative, or -1 if the
        derivative is not in the state vector"""
        raise NotImplementedError('Should be implemented in the derived class '
                                  'as it is benchmark dependent how the state is defined')

    @staticmethod
    def _norm_or_de_norm(vector: np.ndarray, shift: np.ndarray, scale: np.ndarray, de_norm: bool):
        """Normalize or denormalize the state or action.
        :param vector: the state or action to (de)normalize
        :param shift: normalized = (vector - shift) / scale,  de-normalized = vector * scale + shift
        :param scale: normalized = (vector - shift) / scale,  de-normalized = vector * scale + shift
        :param de_norm: transfer from normalized state/action to benchmark specific state/action.
        :return: the (de)normalized state or action"""
        return vector * scale + shift if de_norm else (vector - shift) / scale

    def _reset_log(self) -> None:
        """Notify the loggers that the environment has been reset."""
        for logger in self.loggers:
            logger.reset_log()

    def _step_log_pre(self) -> None:
        """ Logs the state-action (before dynamics function is called)."""
        for logger in self.loggers:
            logger.step_log_pre(self.true_state, self._u)

    def _step_log_post(self, reward: float, terminal: bool) -> None:
        """ Logs the reward and logs sequence based on terminal and reward sum (after dynamics function is called)."""
        for logger in self.loggers:
            logger.step_log_post(self.true_state, reward, terminal)
