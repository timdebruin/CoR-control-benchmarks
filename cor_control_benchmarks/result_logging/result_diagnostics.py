import warnings
from typing import Optional, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from cor_control_benchmarks.control_benchmark import ControlBenchmark, DomainBound
from cor_control_benchmarks.result_logging.trajectory_logging import LogType, TrajectoryLogger


class Diagnostics(object):

    def __init__(self, benchmark: ControlBenchmark, log: LogType, gamma: Optional[float] = None) -> None:
        self.logger = TrajectoryLogger(log=log, gamma=gamma)
        self.benchmark = benchmark
        self.last_diagnostics_episode = 0  # last time diagnostics were asked for (used to give mean of recent episodes)
        benchmark.loggers.append(self.logger)

        colors = cm.get_cmap('tab10')
        self.rewards_color = colors(0)[:-1]
        self.state_color = colors(1)[:-1]
        self.target_color = colors(2)[:-1]
        self.actions_color = colors(3)[:-1]

        self.plots: Dict[
            str, Tuple[plt.Figure, np.ndarray[plt.Axes]]] = {}  # the figures and axes arrays that are
        # displayed, indexed by concatenated names on the y-axes

        plt.ion()

    @property
    def best_reward_sum(self) -> float:
        try:
            return max(self.logger.reward_sum_per_episode)
        except ValueError:
            warnings.warn('Trying to get the best reward sum before any episode has finished, returning some'
                          'low number.', UserWarning)
            return -1e9

    @property
    def last_reward_sum(self) -> float:
        try:
            return self.logger.reward_sum_per_episode[-1]
        except IndexError:
            warnings.warn('Trying to get the last reward sum before any episode has finished, returning some'
                          'low number.', UserWarning)
            return -1e9

    @property
    def best_episode(self) -> int:
        try:
            return int(np.argmax(self.logger.reward_sum_per_episode))
        except ValueError:
            warnings.warn('Trying to get the index of the best episode before any episode has finished, returning -1',
                          UserWarning)
            return -1

    @property
    def last_episode_was_best(self) -> bool:
        return self.best_episode == len(self.logger.reward_sum_per_episode) - 1

    @property
    def highest_observed_return(self) ->float:
        return self.logger.observed_returns['max']

    @property
    def lowest_observed_return(self) -> float:
        return self.logger.observed_returns['min']

    def print_summary(self) -> None:
        print(f'Episodes: {len(self.logger.reward_sum_per_episode)}, best reward sum: {self.best_reward_sum}, '
              f'last reward sum: {self.last_reward_sum}')

    def plot_reward_sum_per_episode(self):
        self._plot_trajectory('reward sum')

    def plot_best_trajectory(self, state: bool, action: bool, rewards: bool):
        self.plot_trajectories(state, action, rewards, episode=self.best_episode)

    def plot_most_recent_trajectory(self, state: bool, action: bool, rewards: bool):
        self.plot_trajectories(state, action, rewards, episode=-1)

    def plot_trajectories(self, state: bool, action: bool, rewards: bool, episode: int):
        if state:
            self._plot_trajectory('state', episode)
        if action:
            self._plot_trajectory('action', episode)
        if rewards:
            self._plot_trajectory('reward', episode)

    def _plot_trajectory(self, name: str = 'state', trajectory_index: Optional[int] = None):
        if name == 'reward sum':
            trajectory = np.array(self.logger.reward_sum_per_episode).reshape(-1, 1)
            target = None
            domain = {
                'max': [0],
                'min': [None]
            }
            time = np.arange(1, len(trajectory) + 1)
            time_name = 'Episode'
            y_names = ['Reward sum']
            plot_color = self.rewards_color
        else:
            assert trajectory_index is not None, 'trajectory index should be given when plotting a specific trajectory'
            t = self.logger.get_trajectory(trajectory_index)
            if t is None:
                warnings.warn('Trying to plot the best episode before any episode has finished', UserWarning)
                return
            if name == 'state':
                trajectory = np.array(t.states)
                target = self.benchmark.target_state
                domain = self.benchmark.not_normalized_state_domain
                y_names = self.benchmark.state_names
                plot_color = self.state_color
            elif name == 'action':
                trajectory = np.array(t.actions)
                target = self.benchmark.target_action
                domain = self.benchmark.not_normalized_action_domain
                y_names = self.benchmark.action_names
                plot_color = self.actions_color
            elif name == 'reward':
                trajectory = np.array(t.rewards).reshape(len(t.rewards), -1)
                target = None
                domain = {
                    'max': [0],
                    'min': [None]
                }
                y_names = ['Reward']
                plot_color = self.rewards_color
            else:
                raise ValueError(f'Unknown trajectory name: {name}')
            time = np.arange(0, len(trajectory)) * self.benchmark.sampling_time
            time_name = 'Time [s]'

        references = np.ones_like(trajectory) * target if target is not None else None

        fig, axes = self.get_fig_and_axes(y_names)
        for component_dim, ax in enumerate(axes):
            ax.clear()
            component_trajectory = trajectory[:, component_dim]
            ax.plot(time, component_trajectory, label=name, color=plot_color)
            if references is not None:
                ax.plot(time, references[:, component_dim], label='reference', color=self.target_color)
                if name == 'state' and self.benchmark.domain_bound_handling[component_dim] == DomainBound.WRAP:
                    ax.plot(time, -references[:, component_dim], label='reference', color=self.target_color)
            d_min = 1.05 * domain['min'][component_dim] if domain['min'][component_dim] is not None else None
            d_max = 1.05 * domain['max'][component_dim] if domain['max'][component_dim] is not None else None
            ax.set_ylim(d_min, d_max)

            ax.set_ylabel(y_names[component_dim])

        plt.xlabel(time_name)
        plt.title(self.benchmark.name)
        plt.legend()
        # plt.draw()
        # plt.show(block=False)
        fig.canvas.draw()

    def get_fig_and_axes(self, y_names):
        key = ','.join(y_names)
        if key not in self.plots:
            f, a = plt.subplots(len(y_names), 1, sharex='col')
            if isinstance(a, plt.Axes):
                a = np.array([a])
            self.plots[key] = f, a
        return self.plots[key]

    def get_scalar_diagnostics_dict(self):
        start = self.last_diagnostics_episode
        end = len(self.logger.reward_sum_per_episode)
        result = {
            'smooth reward sum': np.mean(np.array(self.logger.reward_sum_per_episode[start:end])),
            'max reward sum': self.best_reward_sum,
            'last reward sum': self.last_reward_sum
        }
        self.last_diagnostics_episode = end
        return result
