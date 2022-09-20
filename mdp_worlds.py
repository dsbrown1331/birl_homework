from mdp import FeatureMDP, MDP
import numpy as np
import mdp_utils




def gen_simple_world():
    #four features, blue (true reward of +1), white (true reward of 0), red (true reward of -1), 
    
    b = [1,0,0]  #feature vector for blue colored state
    w = [0,1,0]  #feature vector for white colored state
    r = [0,0,1]  #feature vector for red colored state

    gamma = 0.9  #discount factor

    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, w, 
                      w, r, w,
                      w, w, w]
    feature_weights = [+2.0, -1.0, -3.0] #red feature has weight -1 and blue feature has weight +1 and white feature has weight 0
    
    noise = 0.0 #no noise in transitions
    
    terminals = [0] #set state 0 to be a terminal (absorbing) state, in this implementation, that means there are no outgoing transitions
    
    env = FeatureMDP(3,3,terminals,feature_weights, state_features, gamma, noise)
    
    return env


def gen_test_world():
    #four features, blue (true reward of +1), white (true reward of 0), red (true reward of -1), 
    
    rewards = [1,2,3,4]

    gamma = 0.9  #discount factor

    terminals = [] #set state 0 to be a terminal (absorbing) state, in this implementation, that means there are no outgoing transitions
    
    env = MDP(2, 2, terminals, rewards, gamma)
    
    return env





