# Bayesian IRL Homework

## Dependencies
You will need python (version 3.8 or higher), numpy, and matplotlib. You will not need PyTorch for this assignment.
You should be able to reuse the conda environmnent from Homework 1.

## Assignment

You will need to code two pieces of of a basic implementation of Bayesian IRL. The version of Bayesian IRL provided for you uses Value Iteration given a reward hypothesis to calculate the Values and Q-values. These Q-values are then used for Bayesian IRL's likelihood function. You will gain experience implementing Value Iteration as well as implementing a loglikelihood function for Bayesian IRL.

## Part 1: Code up Value Iteration
Open ```mdp_utils.py``` and fill in the missing code for Value Iteration.

To test your code. run 
```test_value_iteration.py```

Look at ```calculate_q_values``` in ```mdp_utils.py```, to see an example of how to go from a value function to a q-value function. This part is already coded for you.

## Part 2: Code up the loglikelihood function for Bayesian IRL


Open ```bayesian_irl.py``` and read through the ```run_mcmc``` method. This is the part that does MCMC. However, you will need to fill in the missing code for ```def calc_ll()``` using q-values. If you completed part 1, then the q-values implementation is done and you just need to use the q-values when defining the log likelihood. 

To test your code, run
```test_bayesian_irl_ll.py```


## Part 3: Run Bayesian IRL on some demos in a grid world and explore what happens.
If you don't have jupyter notebook installed, you can get it using conda. First activate your conda env ```conda activate imitation_learning``` (assuming  you're using the env from that last assignment). Then run:
```
conda install -c anaconda jupyter
```

For this part you will be using the jupyter notebook ```BayesianIRL.ipynb```. You should be able to run this using the command
```
jupyter notebook BayesianIRL.ipynb
```

## Submission.

Follow the instructions in the jupyter notebook and prepare a pdf report with your answers to the questions and submit the pdf along with your code in a zip file. Submit via Canvas.



