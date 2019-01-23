import warnings
from enum import IntEnum
from typing import List, NamedTuple, Optional
import numpy as np


class LogType(IntEnum):
    """The amount of data that is stored."""
    REWARD_SUM = 1  # only the sum of rewards per episode
    BEST_AND_LAST_TRAJECTORIES = 2  # the states, actions and rewards of the most recent and best episodes
    ALL_TRAJECTORIES = 3  # the states, actions and rewards of all trajectories


class Trajectory(NamedTuple):
    """The state, action and reward trajectories during a single episode."""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]


class TrainingResult(object):
    """Class for storing the relevant parts of a training run, for comparison later"""

    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.episode_reward_sums


class TrajectoryLogger(object):

    def __init__(self, log: LogType) -> None:
        self.log: LogType = log
        self._current_trajectory: Trajectory = Trajectory(states=[], actions=[], rewards=[])
        self.reward_sum_per_episode: List[float] = []
        self.best_trajectory: Optional[Trajectory] = None
        self.last_trajectory: Optional[Trajectory] = None
        self.episode_trajectories: List[Trajectory] = []

    def reset_log(self) -> None:
        """Called when the environment has been reset."""
        if len(self._current_trajectory.rewards) > 0:
            warnings.warn('Episode is reset before the end of the episode was reached, TrajectoryLogger will'
                          'ignore this episode', UserWarning)

    def step_log_pre(self, state: np.ndarray, action: np.ndarray) -> None:
        """ Called during step with the current state and chosen action before dynamics function is called."""
        if self.log > 1:
            self._current_trajectory.states.append(state)
            self._current_trajectory.actions.append(action)

    def step_log_post(self, next_state: np.ndarray, reward: float, terminal: bool) -> None:
        """ Called during step after dynamics function is called with the reward and and terminal."""
        self._current_trajectory.rewards.append(reward)
        if terminal:
            reward_sum_last = sum(self._current_trajectory.rewards)

            if self.log >= LogType.BEST_AND_LAST_TRAJECTORIES:
                self._current_trajectory.states.append(next_state)

                self.last_trajectory = self._current_trajectory
                if reward_sum_last > max(self.reward_sum_per_episode, default=-1e9):
                    self.best_trajectory = self._current_trajectory

                if self.log >= LogType.ALL_TRAJECTORIES:
                    self.episode_trajectories.append(self._current_trajectory)

            self.reward_sum_per_episode.append(reward_sum_last)
            self._current_trajectory = Trajectory(states=[], actions=[], rewards=[])

    def get_trajectory(self, index: int) -> Optional[Trajectory]:
        if self.log == LogType.REWARD_SUM:
            raise ValueError('Trajectory information not stored when using LogType.REWARD_SUM')
        if self.log == LogType.BEST_AND_LAST_TRAJECTORIES:
            try:
                best = np.argmax(self.reward_sum_per_episode)
            except ValueError:
                best = -1
            if index == best:
                return self.best_trajectory
            elif index == -1 or index == len(self.reward_sum_per_episode) - 1:
                return self.last_trajectory
            else:
                raise ValueError(f'Requested trajectory {index}  was neither the best ({best}) nor last '
                                 f'({len(self.reward_sum_per_episode)}). When using '
                                 f'LogType.BEST_AND_LAST_TRAJECTORIES no other trajectories are stored.')
        else:
            return self.episode_trajectories[index]
