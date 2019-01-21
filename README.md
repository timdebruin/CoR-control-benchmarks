# CoR-control-benchmarks
This repository contains python (3.6+) implementations of several control benchmarks. 
These benchmarks require precise control strategies. 
The aim for each is to drive the state of a dynamical system to a reference state. 
Some of these benchmarks (pendulum swing-up, magman) have physical counterparts within the Delft University of Technology.

This readme contains instructions for using this library and references to papers that have used these benchmarks. 

- [Installation and basic usage](#installation-and-basic-usage)
- [Benchmarks and papers that used these benchmarks](#benchmarks-and-papers-that-used-these-benchmarks)
    - [Pendulum swing-up](#pendulum-swing-up)
    - [Magnetic manipulator](#magnetic-manipulator)
    - [Segway](#segway)
    - [Robot navigation](#robot-navigation)
- [Slightly more advanced examples](#advanced-examples)
    - [Logging and plotting results](#logging-and-plotting-results)
    - [Using a state-value function for control](#using-a-state-value-function-for-control)

## Installation and basic usage
Installation / updating:

`pip install --user --upgrade git+git://github.com/timdebruin/CoR-control-benchmarks`

Here the basic usage of the library is shown. More features like logging and plotting results and
using the benchmarks without normalization are shown [later](#advanced-examples).  

```python
import numpy as np
import cor_control_benchmarks as cb

env = cb.MagmanBenchmark(magnets=4)

for episode in range(10):
    terminal = False  # episodes terminate after a time limit or, for some benchmarks, 
    # when the state trajectories leave a predetermined domain
    state = env.reset()  # the environments return a normalized state, with all components 
    # in the range [-1, 1]. Some domains so not enforce this for all state components, but good policies 
    # will stay within this range.

    while not terminal:
        action = np.random.uniform(-1, 1, size=env.action_shape)  # actions should also be normalized in 
        # the domain [-1, 1]  
        state, reward, terminal, _ = env.step(action)
```

## Benchmarks and papers that used these benchmarks
This repository contains the following benchmarks:

### Pendulum swing-up
```python
import cor_control_benchmarks as cb
env = cb.PendulumBenchmark(max_voltage=2.0, sampling_time=0.02, max_seconds=2.5, reward_type=cb.RewardType.QUADRATIC)
```
See [cor_control_benchmarks/pendulum.py](cor_control_benchmarks/pendulum.py) for the dynamics model and available parameters. 

Swing up an under-actuated pendulum. 
To solve the benchmark the pendulum will need to be swung to one side to build momentum before swinging in the opposite 
direction and stabilizing in the upright position.

The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
DCSC / CoR lab.

**Related works**
This benchmark has been used in (among others) the following papers:

- Eduard Alibekov, Jiří Kubalík and Robert Babuška, *"Policy derivation methods for critic-only reinforcement learning in continuous spaces"*, Engineering Applications of Artificial Intelligence (EAAI), 2018 [[paper](https://www.sciencedirect.com/science/article/pii/S0952197617302993), [bibtex](doc/bib/alibekov18-eaai.bib)]
- Tim de Bruin, Jens Kober, Karl Tuyls and Robert Babuška, *"Experience selection in deep reinforcement learning for control"*, Journal of Machine Learning Research (JMLR), 2018 [[paper](http://jmlr.org/papers/v19/17-131.html), [bibtex](doc/bib/debruin2018jmlr.bib)]
- Erik Derner, Jiří Kubalík and Robert Babuška, *"Reinforcement Learning with Symbolic Input-Output Models."*, Conference on Intelligent Robots and Systems (IROS), 2018. [[paper](https://ieeexplore.ieee.org/abstract/document/8593881), [bibtex](doc/bib/derner2018reinforcement.bib)]
- Erik Derner, Jiří Kubalík and Robert Babuška, *"Data-driven Construction of Symbolic Process Models for Reinforcement Learning"*, Conference on Robotics and Automation (ICRA), 2018 [[paper](https://ieeexplore.ieee.org/abstract/document/8461182), [bibtex](doc/bib/derner18icra.bib)]
- Olivier Sprangers, Robert Babuška, Subramanya P. Nageshrao, and Gabriel A. D. Lopes, *"Reinforcement Learning for Port-Hamiltonian Systems"*, IEEE Transactions on Cybernetics, 2015 [[paper](https://ieeexplore.ieee.org/document/6883207), [bibtex](doc/bib/olivier15.bib)]
- Ivo Grondman, Maarten Vaandrager, Lucian Busoniu, Robert Babuška and Erik Schuitema, *"Efficient Model Learning Methods for Actor–Critic Control"*,  Transactions on Systems, Man, and Cybernetics, 2012 [[paper](https://ieeexplore.ieee.org/abstract/document/6096441), [bibtex](doc/bib/grondman12smc.bib)]

### Magnetic manipulator
```python
import cor_control_benchmarks as cb
env = cb.MagmanBenchmark(magnets=4, sampling_time=0.02, max_seconds=2.5, reward_type=cb.RewardType.QUADRATIC)
```
See [cor_control_benchmarks/magman.py](cor_control_benchmarks/magman.py) for the dynamics model and available parameters.

Position a magnetic ball by controlling the current through several electromagnets positioned under a
1-dimensional track that the ball rolls on.

The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
DCSC / CoR lab.

**Related works**
This benchmark has been used in (among others) the following papers:
- Tim de Bruin, Jens Kober, Karl Tuyls and Robert Babuška, *"Experience selection in deep reinforcement learning for control"*, Journal of Machine Learning Research (JMLR), 2018 [[paper](http://jmlr.org/papers/v19/17-131.html), [bibtex](doc/bib/debruin2018jmlr.bib)]
- Eduard Alibekov, Jiří Kubalík and Robert Babuška, *"Policy derivation methods for critic-only reinforcement learning in continuous spaces"*, Engineering Applications of Artificial Intelligence (EAAI), 2018 [[paper](https://www.sciencedirect.com/science/article/pii/S0952197617302993), [bibtex](doc/bib/alibekov18-eaai.bib)]
- Jan-Willem Damsteeg, Subramanya P. Nageshrao, and Robert Babuška, *"Model-based real-time control of a magnetic manipulator system"*, Conference on Decision and Control (CDC), 2017 [[paper](https://ieeexplore.ieee.org/document/8264140), [bibtex](doc/bib/damsteeg17.bib)]

### Segway
Model of a Segway. The segway starts out under a significant angle and the control challenge is to right it
quickly and stabilize.

```python
import cor_control_benchmarks as cb
env = cb.SegwayBenchmark(sampling_time=0.01, max_seconds=2.5, reward_type=cb.RewardType.ABSOLUTE)
```
See [cor_control_benchmarks/segway.py](cor_control_benchmarks/segway.py) for the dynamics model and available parameters.

### Robot navigation
Drive a navigation robot to a target position and have it stand there facing the correct direction.

```python
import cor_control_benchmarks as cb
env = cb.SegwayBenchmark(sampling_time=0.01, max_seconds=2.5, reward_type=cb.RewardType.ABSOLUTE)
```
See [cor_control_benchmarks/segway.py](cor_control_benchmarks/segway.py) for the dynamics model and available parameters.


## Advanced examples
This section contains (slightly more) advanced usage examples.

### Logging and plotting results
Some basic logging and plotting functions come with the library. To use these, make a `Diagnostics` object, giving it the
benchmark instance for which we want to log the trajectories and the amount of information that should be stored. The logging options are:
```python
import cor_control_benchmarks as cb
log = cb.LogType.REWARD_SUM  # only store the sum of rewards for every episode
log = cb.LogType.BEST_AND_LAST_TRAJECTORIES  # store the states, actions and rewards at every time step of both the most recent and the best episode, as well as the sum of rewards for every episode 
log = cb.LogType.ALL_TRAJECTORIES  #  # store the states, actions and rewards at every time step of every episode
```
Based on what is logged, different plots are available:
```python
import cor_control_benchmarks as cb
import numpy as np

env = cb.PendulumBenchmark(max_voltage=3.)
diagnostics = cb.Diagnostics(benchmark=env, log=cb.LogType.ALL_TRAJECTORIES)

for episode in range(10):
    terminal = False  
    state = env.reset()  

    while not terminal:
        action = np.random.uniform(-1, 1, size=env.action_shape)  
        state, reward, terminal, _ = env.step(action)
    diagnostics.print_summary() # print to the terminal the number of episodes that have passed, the best reward sum so far and the most recent reward sum (works with all log types)
    diagnostics.plot_reward_sum_per_episode() # Give a plot of the learning curve (works with all log types)

diagnostics.plot_most_recent_trajectory(state=True, action=True, rewards=True) # plot the states, actions and/or reward trajectories during the most recent episode (works with LogType.BEST_AND_LAST_TRAJECTORIES and LogType.ALL_TRAJECTORIES)
diagnostics.plot_best_trajectory(state=True, action=True, rewards=True) # plot the states, actions and/or reward trajectories during the episode with the highest reward sum so far (works with LogType.BEST_AND_LAST_TRAJECTORIES and LogType.ALL_TRAJECTORIES)
diagnostics.plot_trajectories(state=True, action=True, rewards=True, episode=3) # plot the states, actions and/or reward trajectories during a specific episode (works only with LogType.ALL_TRAJECTORIES)

input('Press enter to close figures')  # Since the figures do not block, having the script terminate would close them
```

### Using a state-value function for control
Assume we have a function that tells us for a given state (in the non normalized state domain) the state-value. To use this for control we can tell the benchmarks not to use normalization. 

To select the (approximately) best action from a state we can repeatedly reset the environment to that specific state and see what state we end up in for the actions we consider.

To still use the logging functions described above we can make two instances of the benchmark: the one we keep resetting (which would interfere with logging) and one that we actually perform the rollouts on.

````python
import cor_control_benchmarks as cb

rollout_pendulum = cb.PendulumBenchmark(max_voltage=2., do_not_normalize=True)
dynamics_check_pendulum = cb.PendulumBenchmark(max_voltage=2., do_not_normalize=True)

diagnostics = cb.Diagnostics(benchmark=rollout_pendulum, log=cb.LogType.BEST_AND_LAST_TRAJECTORIES)

def next_state_from(initial_state, action):
    dynamics_check_pendulum.reset_to_specific_state(initial_state)
    next_state, _, _, _, = dynamics_check_pendulum.step(action)
    return next_state

state = rollout_pendulum.reset()
terminal = False
reward_sum = 0
while terminal is not True:
    best_value, best_action = -1e6, 0.
    for action_to_consider in [-2., -1., 0., 1., 2.]:
        v = state_value_function(next_state_from(state, action_to_consider))
        if v > best_value:
            best_value = v
            best_action = action_to_consider

    state, _, _, _ = rollout_pendulum.step(best_action)
   
diagnostics.plot_most_recent_trajectory(state=True, action=True, rewards=True)
````
 
   
