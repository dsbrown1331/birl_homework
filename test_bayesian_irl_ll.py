from distutils import log
import bayesian_irl
import numpy as np

from mdp_worlds import gen_simple_world

env = gen_simple_world()

#test 1
demo = [(2,1)]
birl = bayesian_irl.BIRL(env, demo, beta=10)
hypothesis_reward = np.array([+1,0,-1])
log_likelihood = birl.calc_ll(hypothesis_reward)
print("ll", log_likelihood)
solution = -0.7781844091508692
error = abs(solution - log_likelihood)
# print(error)


if error < 0.0001:
    print("#"*20)
    print("Solution verified for test 1.")
    print("#"*20)
else:
    print("#"*20)
    print("Incorrect. Please try again")
    print("Correct loglikelihood is", solution, "but you returned", log_likelihood)
    print("#"*20)


#test 2
demo = [(4,0)]
birl = bayesian_irl.BIRL(env, demo, beta=1)
hypothesis_reward = np.array([+1,+1,-1])
log_likelihood = birl.calc_ll(hypothesis_reward)
solution = -2.952247486594
error = abs(solution - log_likelihood)
# print(error)


if error < 0.0001:
    print("#"*20)
    print("Solution verified for test 2. Continue to Part 3")
    print("#"*20)
else:
    print("#"*20)
    print("Incorrect. Please try again")
    print("Correct loglikelihood is", solution, "but you returned", log_likelihood)
    print("#"*20)

