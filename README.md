# CoR-control-benchmarks
This repository contains python (3.6+) implementations of several control benchmarks. 
These benchmarks require precise control strategies. 
The aim for each is to drive the state of a dynamical system to a reference state. 
Some of these benchmarks (pendulum swingup, magman) have physical counterparts within the Delft University of Technology.

This repository (will) contain(s) both these benchmarks as well as several implementations of algorithms that solve these tasks.
 

This readme contains instructions for using the benchmarks, and references to papers that have used these benchmarks. 

## Usage
The file [example.py](example.py) contains examples of using these benchmarks and the helper code available. 
Here the most basic usage (enough for trial based control) is shown:  

```python
import numpy as np
import cor_control_benchmarks as cb

env = cb.magman.MagmanBenchmark(magnets=4)

for episode in range(10):
    terminal = False  # episodes terminate after a time limit or, for some benchmarks, 
    # when the state trajectories leave a predetermined domain
    state = env.reset()  # the environments return a normalized state, with all components 
    # in the range [-1, 1]. Some domains so not enforce this for all state components, but good policies 
    # will stay within this range.

    while not terminal:
        action = np.random.uniform(low=-1., high=1., size=(4,))  # actions should also be normalized in 
        # the domain [-1, 1]  
        state, reward, terminal, _ = env.step(action)
```

## Benchmarks and papers that used these benchmarks
This repository contains the following benchmarks:

### Pendulum swing-up
```python
import cor_control_benchmarks as cb
env = cb.pendulum.PendulumBenchmark(max_voltage=2.0, sampling_time=0.02, max_seconds=2.5, reward_type=cb.control_benchmark.RewardType.QUADRATIC)
```
See [benchmarks/pendulum.py](cor_control_benchmarks/pendulum.py) for the dynamics model and available parameters. 

Swing up an under-actuated pendulum. 
To solve the benchmark the pendulum will need to be swung to one side to build momentum before swinging in the opposite 
direction and stabilizing in the upright position.

The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
DCSC / CoR lab.

**Related works**
This benchmark has been used in (among others) the following papers:

- Erik Derner, Jiří Kubalík and Robert Babuška, *"Reinforcement Learning with Symbolic Input-Output Models."*, Conference on Intelligent Robots and Systems (IROS), 2018. [[paper](https://ieeexplore.ieee.org/abstract/document/8593881), [bibtex](doc/bib/derner2018reinforcement.bib)]
- Tim de Bruin, Jens Kober, Karl Tuyls and Robert Babuška, *"Experience selection in deep reinforcement learning for control"*, Journal of Machine Learning Research (JMLR), 2018 [[paper](http://jmlr.org/papers/v19/17-131.html), [bibtex](doc/bib/debruin2018jmlr.bib)]
- Erik Derner, Jiří Kubalík and Robert Babuška, *"Data-driven Construction of Symbolic Process Models for Reinforcement Learning"*, Conference on Robotics and Automation (ICRA), 2018 [[paper](https://ieeexplore.ieee.org/abstract/document/8461182), [bibtex](doc/bib/derner18icra.bib)]
- Olivier Sprangers, Robert Babuška, Subramanya P. Nageshrao, and Gabriel A. D. Lopes, *"Reinforcement Learning for Port-Hamiltonian Systems"*, IEEE Transactions on Cybernetics, 2015 [[paper](https://ieeexplore.ieee.org/document/6883207), [bibtex](doc/bib/olivier15.bib)]
- Ivo Grondman, Maarten Vaandrager, Lucian Busoniu, Robert Babuška and Erik Schuitema, *"Efficient Model Learning Methods for Actor–Critic Control"*,  Transactions on Systems, Man, and Cybernetics, 2012 [[paper](https://ieeexplore.ieee.org/abstract/document/6096441), [bibtex](doc/bib/grondman12smc.bib)]

### Magnetic manipulator
```python
import cor_control_benchmarks as cb
env = cb.magman.MagmanBenchmark(magnets=4, sampling_time=0.02, max_seconds=2.5, reward_type=cb.control_benchmark.RewardType.QUADRATIC)
```
See [benchmarks/magman.py](cor_control_benchmarks/pendulum.py) for the dynamics model and available parameters.

Position a magnetic ball by controlling the current through several electromagnets positioned under a
1-dimensional track that the ball rolls on.

The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
DCSC / CoR lab.

**Related works**
This benchmark has been used in (among others) the following papers:
- Tim de Bruin, Jens Kober, Karl Tuyls and Robert Babuška, *"Experience selection in deep reinforcement learning for control"*, Journal of Machine Learning Research (JMLR), 2018 [[paper](http://jmlr.org/papers/v19/17-131.html), [bibtex](doc/bib/debruin2018jmlr.bib)]
- Jan-Willem Damsteeg, Subramanya P. Nageshrao, and Robert Babuška, *"Model-based real-time control of a magnetic manipulator system"*, Conference on Decision and Control (CDC), 2017 [[paper](https://ieeexplore.ieee.org/document/8264140), [bibtex](doc/bib/damsteeg17.bib)]

### Segway
Model of a Segway. The segway starts out under a significant angle and the control challenge is to right it
quickly and stabilize.

```python
import cor_control_benchmarks as cb
env = cb.segway.SegwayBenchmark(sampling_time=0.01, max_seconds=2.5, reward_type=cb.control_benchmark.RewardType.ABSOLUTE)
```
See [benchmarks/magman.py](cor_control_benchmarks/segway.py) for the dynamics model and available parameters.

