from distutils import log
import bayesian_irl
import numpy as np

from mdp_worlds import gen_simple_world

env = gen_simple_world()

beta = 10.0       #assume near optimal demonstrator
demo = [(2,1)]
birl = bayesian_irl.BIRL(env, demo, beta)
hypothesis_reward = np.array([+1,+1,-10])
log_likelihood = birl.calc_ll(hypothesis_reward)
solution = -1.0986122886681073
error = abs(solution - log_likelihood)
print(error)


if error < 0.0001:
    print("#"*20)
    print("Solution verified. Continue to Part 3")
    print("#"*20)
else:
    print("#"*20)
    print("Incorrect. Please try again")
    print("Correct loglikelihood is", solution, "but you returned", log_likelihood)
    print("#"*20)


