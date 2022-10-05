from mdp_utils import value_iteration
from mdp_worlds import gen_test_world, gen_test_world2
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
    print("Solution verified for test #1.")
    print("#"*20)
else:
    print("#"*20)
    print("Incorrect. Please try again")
    print("Your value iteration returned")
    print(values)
    print("but should have returned")
    print(solution)
    print("#"*20)

#second test
env2 = gen_test_world2()

values2 = value_iteration(env2, epsilon=eps)
print(values2)

solution2 = np.array([ 1.,  -2.04957, -3.1545, -6.4722])
error = max(np.abs(values2 - solution2))
correct = error <= eps

if correct:
    print("#"*20)
    print("Solution verified for test #2. Continue to Part 2")
    print("#"*20)
else:
    print("#"*20)
    print("Incorrect. Please try again")
    print("Your value iteration returned")
    print(values2)
    print("but should have returned")
    print(solution2)
    print("#"*20)