from mdp import MDP, FeatureMDP
from matplotlib import pyplot as plt
import time
import numpy as np
import math
import copy


def value_iteration(env, epsilon=0.0001):
    """
  :param env: the MDP
  :param epsilon: numerical precision for value function
  :return: vector representation of the value function for each state
  """
    num_s = env.num_states
    num_a = env.num_actions
    V = np.zeros(num_s)  #vector to store Value function


    ############################
    ##TODO implement Value Iteration such that it terminates when within epsilon of the true Value function
    ## Implement Value Iteration basd on the Russell and Norvig Chapter on MDPs (page 653 in the 3rd Edition). 
    ## This chapter is included in Supplemental materials for day 1 on the class website.
    ## Note the following: 
    ## env.transitions[s][a] will give you a vector of transition probabilities, one for each possible next state.
    ## env.transitions[s][a][s2] would give you the probability of ending up in state s2 after taking action a in state s.
    ## env.rewards will give you a vector of rewards for all states and env.rewards[s] will give you the reward at a particular state s.
    ## States are indexed 0 to env.num_states-1 and are numbered left to right, top to bottom.
    ## Finally, env.gamma gives you the discount factor.
    ############################
    
    return V


def get_optimal_policy(env, epsilon=0.0001, V=None):
    #runs value iteration if not supplied as input
    if not V:
        V = value_iteration(env, epsilon)
    n = env.num_states
    optimal_policy = []  # our game plan where we need to

    for s in range(n):
        max_action_value = -math.inf
        best_action = 0

        for a in range(env.num_actions):
            action_value = 0.0
            for s2 in range(n):  # look at all possible next states
                action_value += env.transitions[s][a][s2] * V[s2]
                # check if a is max
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = a  # direction to take
        optimal_policy.append(best_action)
    return optimal_policy


def logsumexp(x):
    max_x = np.max(x)
    sum_exp = 0.0
    for xi in x:
        sum_exp += np.exp(xi - max_x)
    return max(x) + np.log(sum_exp)



def calculate_q_values(env, V=None, epsilon=0.0001):
    """
  gets q values for a markov decision process

  :param env: markov decision process
  :param epsilon: numerical precision
  :return: reurn the q values which are
  """

    #runs value iteration if not supplied as input
    if not V:
        V = value_iteration(env, epsilon)
    n = env.num_states

    Q_values = np.zeros((n, env.num_actions))
    for s in range(n):
        for a in range(env.num_actions):
            Q_values[s][a] = env.rewards[s] + env.gamma * np.dot(env.transitions[s][a], V)

    return Q_values




def action_to_string(act, UP=0, DOWN=1, LEFT=2, RIGHT=3):
    if act == UP:
        return "^"
    elif act == DOWN:
        return "v"
    elif act == LEFT:
        return "<"
    elif act == RIGHT:
        return ">"
    else:
        return NotImplementedError


def visualize_trajectory(trajectory, env):
    """input: list of (s,a) tuples and mdp env
        ouput: prints to terminal string representation of trajectory"""
    states, actions = zip(*trajectory)
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in states:
                #get index
                indx = states.index(count)
                if count in env.terminals:
                    policy_row += ".\t"    
                else:    
                    policy_row += action_to_string(actions[indx]) + "\t"
            else:
                policy_row += " \t"
            count += 1
        print(policy_row)



def visualize_policy(policy, env):
    """
  prints the policy of the MDP using text arrows and uses a '.' for terminals
  """
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in env.terminals:
                policy_row += ".\t"    
            else:
                policy_row += action_to_string(policy[count]) + "\t"
            count += 1
        print(policy_row)


def print_array_as_grid(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{:.2f}\t".format(array_values[count])
            count += 1
        print(print_row)


def print_array_as_grid_raw(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{}\t".format(array_values[count])
            count += 1
        print(print_row)


def visualize_feature_reward(feature_weights, env):
    assert type(env) == FeatureMDP
    r = []
    for feature_vector in env.state_features:
        r.append(np.dot(feature_vector,feature_weights))
    print_array_as_grid(r,env)

def sample_l2_ball(k):
    #sample a vector of dimension k with l2 norm of 1
    sample = np.random.randn(k)
    return sample / np.linalg.norm(sample)