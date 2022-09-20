from mdp_utils import value_iteration
from mdp_worlds import gen_test_world
import numpy as np

#create simple MDP
env = gen_test_world()
eps=0.0001  #desired numerical precision

values = value_iteration(env, epsilon=eps)
print(values)

solution = np.array([34.21558,  36.22837, 37.32727, 38.73392])
error = max(np.abs(values - solution))
correct = error <= eps

if correct:
    print("#"*20)
    print("Solution verified. Continue to Part 2")
    print("#"*20)
else:
    print("#"*20)
    print("Incorrect. Please try again")
    print("Your value iteration returned")
    print(values)
    print("but should have returned")
    print(solution)
    print("#"*20)

