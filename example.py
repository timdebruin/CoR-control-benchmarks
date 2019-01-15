import random

from benchmarks.pendulum import PendulumBenchmark

pendulum = PendulumBenchmark()

for episode in range(1):
    reward_sum = 0
    terminal = False
    state = pendulum.reset()

    while not terminal:
        action = random.uniform(-1, 1)
        state, reward, terminal, _ = pendulum.step(action)
        reward_sum += reward
        print(state, reward)
        print(pendulum.denormalize_action(action))

    print(f'reward sum: {reward_sum}')
