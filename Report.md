# Report

## Learning Algorithm

The algorithm implemented to solve this environment is Deep Deterministic Policy Gradient which combines both Q-learning and Policy gradients. At its core, it uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn.This is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.


## The Model
 It primarily uses two neural networks, one for the actor and one for the critic. The critic is a Q-value network that takes in state and action as input and outputs the Q-value. 
 
 ![](images/NeuralNetwork.png)


* The Critic-Network consists of two hiddenlayer with Sigmoid activation.

* The Actor-Network consists of two hiddenlayers with relu activation. For the Output-layer a tanh-function is used in order to map the 	output the servo angles.

The following pseudocode shows the DDPG Algorithm by (Lillicrap et al., 2015).
 ![](images/pseudocode)






## Architecture



## Result




## Future Improvement


## Acknowledgments


