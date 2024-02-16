# Description

My reimplementation of various Deep reinforcement learning algorithms such as [Deep Q-learning](https://arxiv.org/abs/1312.5602), [Double Deep Q-learning](https://arxiv.org/pdf/1509.06461.pdf) and [Clipped Double Deep Q-Learning](https://arxiv.org/pdf/1802.09477.pdf). The algorithms are then used to learn how to play a variation of Flappy Bird. The evaluation can be found in the [evaluation](#evaluation) section.

https://github.com/NicolaWidmer/FlappyAI/raw/main/media/example.mp4

Video of Clipped Double Deep Q-Learning playing Flappy Bird after 600 episodes of training.

# Requirements

To install the requirements run
```
pip install -r requirements.txt
```

# Usage


## Arguments

```
usage: main.py [-h] [-m {DQN,DDQN,CDDQN}] [-lr LR] [-ls LAYER_SIZE] [-g GAMMA] [-e EPSILON] [-t TAU] [-n EPISODES] [-s SEED] [-v VIDEO] [-c] [-p]

FlappyAI is a programm which can usedifferent AI algorithms to learn how to play flappy bird

options:
  -h, --help           show this help message and exit
  -m {DQN,DDQN,CDDQN}  The alorithm you want to train the options are DQN for Deep Q-Network, DDQN for Double Deep Q-Network, CDDQN for Clipped
                       Double Deep Q-Network
  -lr LR               The learning rate of the gradient descent
  -ls LAYER_SIZE       The hidden layer size of the neural networks used by the alorithms
  -g GAMMA             The discount factor of the reward
  -e EPSILON           Epsilon is used in the training of the AI's and describes the probability of takeing a random action
  -t TAU               The update factor for the target network
  -n EPISODES          The number of episodes to train
  -s SEED              The seed for the random generators
  -v VIDEO             The path to where to video of the result should be stored
  -c                   Run all algorithms and compare them
  -p                   Play the game yourself
```
## Example

To run Double Deep Q-Network for 500 episodes run.
```
python main.py -m DDQN -n 500
```


# Evaluation

The plot was created by running
```
python main.py -c -v mp4s
```
and therfore using the default parameters.

![Comparison](./media/comparison.png)

The trian/evaluation rewards plotted are averaged over the last 100 episodes otherwise the plot would be much more noisy. The gap between training and evaluation reward happens because during training with probability $\epsilon$ a random action is chosen, which may not be optimal. One can also see the typical problem of classical Deep Q-learning after some time, it gets overconfident about the learned Q-Value and begins to perform worse. Things which Double Deep Q-learning and Clipped Double Deep Q-Learning fix by some extent. 

Below are videos of one episode during training for different algorithms and different number of train episodes.

### DQN

After 200 episodes
<video width="800" height="600" src="media/DQN_episode200.mp4" autoplay loop></video>

After 400 episodes
<video width="800" height="600" src="media/DQN_episode600.mp4" autoplay loop></video>

After 800 episodes
<video width="800" height="600" src="media/DQN_episode1000.mp4" autoplay loop></video>

### Double DQN

After 200 episodes
<video width="800" height="600" src="media/DoubleDQN_episode200.mp4" autoplay loop></video>

After 400 episodes
<video width="800" height="600" src="media/DoubleDQN_episode600.mp4" autoplay loop></video>

After 800 episodes
<video width="800" height="600" src="media/DoubleDQN_episode1000.mp4" autoplay loop></video>

### Clipped Double DQN

After 200 episodes
<video width="800" height="600" src="media/ClippedDoubleDQN_episode200.mp4" autoplay loop></video>

After 400 episodes
<video width="800" height="600" src="media/ClippedDoubleDQN_episode600.mp4" autoplay loop></video>

After 800 episodes
<video width="800" height="600" src="media/ClippedDoubleDQN_episode1000.mp4" autoplay loop></video>