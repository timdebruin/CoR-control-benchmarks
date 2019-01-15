from cor_control_benchmarks.pendulum import PendulumBenchmark
from cor_control_benchmarks.magman import MagmanBenchmark
from cor_control_benchmarks.segway import SegwayBenchmark
from cor_control_benchmarks.robot_navigation import RobotNavigationBenchmark

from cor_control_benchmarks.control_benchmark import RewardType, DomainBound

name = "cor_control_benchmarks"
__all__ = ["control_benchmark", "magman", "segway", "pendulum", "robot_navigation", "result_logging"]
