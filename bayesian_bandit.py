import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75] # initial click through rates


class Bandit(object):# acts like a slot machine
  def __init__(self, p):
    self.p = p # probability of winning
    # Beta parameters which are both 1, aka a uniform distribution.
	self.a = 1
    self.b = 1

  def pull(self):
    return np.random.random() < self.p # if its less than p, we get 1, otherwise 0.

  def sample(self):
    return np.random.beta(self.a, self.b) # sample from its current Beta distribution.

  def update(self, x): # function to update a and b. -> x is either 0 or 1.
    self.a += x
    self.b += 1 - x

# plot the pdf of each bandit to compare them on the same chart.
def plot(bandits, trial): # plot each bandit, trial is trial number.
  x = np.linspace(0, 1, 200) # we know its between 0 and 1, lets give it 200 points, so it looks reasonably smooth.
  for b in bandits:
    y = beta.pdf(x, b.a, b.b)
    plt.plot(x, y, label="real p: %.4f" % b.p)
  plt.title("Bandit distributions after %s trials" % trial)
  plt.legend()
  plt.show()


def experiment():
  # init array of bandits.
  # each has its own p.
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

  # these are the points we are going to show a plot.
  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
  for i in xrange(NUM_TRIALS):# loop through each trial

    # take a sample from each bandit
    bestb = None # best bandit, the bandit whose arm we eventually pull.
    maxsample = -1 # keep track of the maximum sample we got.
    allsamples = [] # let's collect these just to print for debugging
    for b in bandits:
      sample = b.sample()
      allsamples.append("%.4f" % sample)
      if sample > maxsample:
        maxsample = sample
        bestb = b
    if i in sample_points:
      print "current samples: %s" % allsamples
      plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bestb.pull()

    # update the distribution for the bandit whose arm we just pulled
    bestb.update(x)


if __name__ == "__main__":
  experiment()
