import matplotlib.pyplot as plt
import numpy as np
from bayesian_bandit import Bandit
"""lets do a demo to see how the click through rate, when we 
use the bayesian_bandit method."""

def run_experiment(p1, p2, p3, N): # 3 probabilities = 3 bandits and number of trials.
  # define each bandit
  bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]

  # keep track of all the data we get. 1 for click, 0 for no-click
  data = np.empty(N)
  
  for i in xrange(N):
    # thompson sampling
    j = np.argmax([b.sample() for b in bandits])
    x = bandits[j].pull()
    bandits[j].update(x)

    # for the plot, keep track of x which is the data at i.
    data[i] = x
  
  cumulative_average_ctr = np.cumsum(data) / (np.arange(N) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average_ctr)
  plt.plot(np.ones(N)*p1)
  plt.plot(np.ones(N)*p2)
  plt.plot(np.ones(N)*p3)
  plt.ylim((0,1))
  plt.xscale('log')
  plt.show()


run_experiment(0.2, 0.25, 0.3, 100000)