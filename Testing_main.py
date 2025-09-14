import numpy as np
from matplotlib import pyplot as plt
from SpecializedTeams import SpecializedTeams, norm, normalize
from OptimizationModel import OptimizationModel
from time import perf_counter_ns as timer


plt.style.use("ggplot")


def QuinticOpt(x):
  x2 = x * x
  x3 = x * x * x
  x4 = x3 * x
  x5 = x4 * x
  return np.abs(x5 - 3.0 * x4 + 4.0 * x3 + 2.0 * x2 - 10.0 * x - 4.0).sum()

def GrieOpt(x):
  #x = np.round(x)
  return 0.00025 * (x * x).sum() + np.cos(x / np.sqrt(1.0 + np.arange(len(x)))).prod() + 1.0

def MichOpt(x):
  m = np.arange(len(x)) + 1.0
  return -(np.sin(x) * (np.sin(m * x * x / np.pi) ** 20)).sum()

def AlpOpt(x):
  return np.abs(x * np.sin(x) + 0.1 * x).sum()

def TwoDimCircleEnergyMinimizer(thetas):
  x1 = np.array((1.0, 0.0))
  theta1, theta2 = thetas
  x2 = np.hstack((np.sin(theta1), np.cos(theta1),))
  x3 = np.hstack((np.sin(theta2), np.cos(theta2),))
  d1, d2, d3 = x1 - x2, x2 - x3, x3 - x1
  return 10.0 * np.exp(-0.01 * np.sqrt((d1 * d1 + d2 * d2 + d3 * d3).sum()))

def MyTestFunc(x):
    return -(np.exp(-0.1 * (x * x).sum()) + 1.5 * np.exp(-7.5 * ((x - np.array((3.0, 4.0,))) ** 2).sum()) + 1.55 * np.exp(-0.5 * (x + np.array((1.0, 5.0))) ** 2).sum())


if __name__ == "__main__":
    test_dim = 5
    Test_Model = OptimizationModel(test_dim)


    Test_Model.SetObjective(GrieOpt)
    Test_Model.SetConstraints()

    Test_Model.StartOptimization(100.0, team_size=20, team_number=20)


