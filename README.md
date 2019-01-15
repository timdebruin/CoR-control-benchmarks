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
from benchmarks.magman import MagmanBenchmark

env = MagmanBenchmark(magnets=4)

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
from benchmarks.pendulum import PendulumBenchmark
from benchmarks.control_benchmark import RewardType
env = PendulumBenchmark(sampling_time=0.02, max_seconds=2.5, reward_type=RewardType.QUADRATIC)
```
See [benchmarks/pendulum.py](benchmarks/pendulum.py) for the dynamics model and available parameters. 

Swing up an under-actuated pendulum. 
To solve the benchmark the pendulum will need to be swung to one side to build momentum before swinging in the opposite 
direction and stabilizing in the upright position.

The parameters for the dynamics are based on a physical setup present in the Delft University of Technology
DCSC / CoR lab.

This benchmark has been used in (possibly among others) the following papers:

- Reinforcement Learning with Symbolic Input-Output Models, *Derner et. al*, 2018 ([paper](https://ieeexplore.ieee.org/abstract/document/8593881), [bibtex](doc/bib/derner2018reinforcement.md))
- 

### Magnetic manipulator


### Segway



