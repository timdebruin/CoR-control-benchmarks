from typing import List, NamedTuple, Optional
import numpy as np


class Trajectory(NamedTuple):
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]


class TrajectoryLogger(object):

    def __init__(self):
        self.current_trajectory: Trajectory = Trajectory(states=[], actions=[], rewards=[])
        self.best_trajectory: Optional[Trajectory] = None

    def reset_log(self):
        raise NotImplementedError

    def step_log_pre(self, state: np.ndarray, action: np.ndarray):
        raise NotImplementedError

    def step_log_post(self, reward: float, terminal: bool):
        raise NotImplementedError


class DummyLogger(TrajectoryLogger):

    def __init__(self):
        super().__init__()

    def reset_log(self):
        pass

    def step_log_pre(self, state: np.ndarray, action: np.ndarray):
        pass

    def step_log_post(self, reward: float, terminal: bool):
        pass

