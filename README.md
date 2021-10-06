# DRLND-continous-learning

### Table of Contents

1. [Summary](#summary)
2. [Installation](#installation)
3. [Getting Started](#Getting Started)
4. [File Descriptions](#files)
5. [Experiments](#experiments)

### Summary <a name="summary"></a>

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Distributed Training
For this project, we will provide you with two separate versions of the Unity environment:

The first version contains a single agent.
The second version contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Environment <a name="Environment"></a>

* Set-up: Double-jointed arm which can move to target locations.
* Goal: Each agent must move its hand to the goal location, and keep it there.
* Agents: The environment contains 20 agents linked to a single Brain.
* Agent Reward Function (independent):
* +0.1 for each timestep agent's hand is in goal location.
*Brains: One Brain with the following observation/action space.
         * Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
         * Vector Action space: (Continuous) Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
           Every entry in the action vector should be a number between -1 and 1.
         * Visual Observations: None.
* Reset Parameters: Two, corresponding to goal size, and goal movement speed.
* Environment Solving Criteria: The target for the agent is to solve the environment by achieving a score of +30 averaged across all 20 agents for 100 consecutive episodes.

#### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment.

Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.


## Installation <a name="installation"></a>

This code has been tested with Anaconda distribution of `Python 3.6`. Additional libraries used in the project are: 

cudatoolkit 10.0<br>
numpy 1.19.1: Install with 'pip install numpy'. <br> 
matplotlib 3.2.2<br>
ml-agents: Install by following instructions here.
Beyond performing standard installation of the above packages, no additional installations are required to run code in this project.

## Getting Started <a name="Getting Started"></a>


Download the environment from one of the links below. You need only select the environment that matches your operating system:

Version 1: One (1) Agent

* Linux: click here
* Mac OSX: click here
* Windows (32-bit): click here
* Windows (64-bit): click here

Version 2: Twenty (20) Agents

* Linux: click here
* Mac OSX: click here
* Windows (32-bit): click here
* Windows (64-bit): click here
* 
(For Windows users) Check out this link(https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen (https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-on-Amazon-Web-Service.md)), then please use this link (version 1) or this link (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen(https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

## File Descriptions <a name="files"></a>

The code is structured as follows:

* Continuous_Control.ipynb: This is where the DDPG agent is trained.
* Continuous_Control.html: html view for Continuous_Control.ipynb
* ddpg_agent.py: This module implements a class to represent a DDPG agent.
* model.py: This module contains the implementation of the Actor and Critic neural networks.
* checkpoint_actor.pth: This is the binary containing the trained neural network weights for Actor.
* checkpoint_critic.pth: This is the binary containing the trained neural network weights for Critic.
* Report.md: Project report and result analysis.
* README.md: Readme file.


## Experiments <a name="experiments"></a>

Follow the instructions in Continuous_Control.ipynb to get started with training your own agent!

Trained model weights is included for quickly running the agent and seeing the result in Unity ML Agent.

Run the last cell of the notebook Continuous_Control.ipynb.


