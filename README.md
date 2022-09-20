# Bayesian IRL Homework

You will need to code two pieces of Bayesian IRL. We will be implementing a simple version of Bayesian IRL that uses Value Iteration given a reward hypothesis to calculate the Values and Q-values. This is important since Bayesian IRL's likelihood function is based on Q-values.

## Part 1: Code up Value Iteration
Open ```mdp_utils.py``` and fill in the missing code for Value Iteration.

To test your code. run 
```test_value_iteration.py```

## Part 2: Code up the loglikelihood function for Bayesian IRL

Open ```bayesian_irl.py``` and fill in the missing code for ```def calc_ll()```
To test your code, run
```test_bayesian_irl_ll.py```

## Part 3: Run Bayesian IRL on some demos in a grid world and explore what happens.
For this part you will be using the jupyter notebook ```BayesianIRL.ipynb```. You should be able to run this using the command
```
jupyter notebook BayesianIRL.ipynb
```
Follow the instructions and prepare a pdf report with your answers to the questions and submit the pdf along with your code in a zip file.



