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
The critic is trained by minimizing the bellman equation. But in contrast to Deep-Q-Learning it only outpus one Q-value per state-action pair. The actor on the other hand can be trained by directly applying the gradient. The equation was derived by (Silver et al., 2014).

However enviornment, actions, state and reward need to be defined:

#### The Environment

**Set-up:** Double-jointed arm which can move to target locations.
**Goal:** Each agent must move its hand to the goal location, and keep it there.
**Agents:** The environment contains 20 agents linked to a single Brain.
Agent Reward Function (independent):
+0.1 for each timestep agent's hand is in goal location.
**Brains:** One Brain with the following observation/action space.
* Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
* Vector Action space: (Continuous) Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
* Visual Observations: None.
**Reset Parameters:** Two, corresponding to goal size, and goal movement speed.
**Environment Solving Criteria:** The target for the agent is to solve the environment by achieving a score of +30 averaged across all 20 agents for 100 consecutive episodes.
The most straigh forward approach is to define the actions by a twelve dimensional vecto






## Architecture



## Result




## Future Improvement


## Acknowledgments


