# Description

My reimplementation of various Deep reinforcement learning algorithms such as [Deep Q-learning](https://arxiv.org/abs/1312.5602), [Double Deep Q-learning](https://arxiv.org/pdf/1509.06461.pdf) and [Clipped Double Deep Q-Learning](https://arxiv.org/pdf/1802.09477.pdf). The algorithms are then used to learn how to play a variation of Flappy Bird. The evaluation can be found in the [evaluation](#evaluation) section.



https://github.com/NicolaWidmer/FlappyAI/assets/61154523/e8261f49-c21e-42ea-9bd0-4855995f3497


Video of Clipped Double Deep Q-Learning playing Flappy Bird after 600 episodes of training.

# Requirements

To install the requirements run
```
pip install -r requirements.txt
```

# Usage


```
usage: main.py [-h] [-m {DQN,DDQN,CDDQN}] [-lr LR] [-ls LAYER_SIZE] [-g GAMMA] [-e EPSILON] [-t TAU] [-n EPISODES] [-s SEED] [-v VIDEO] [-c] [-p]

FlappyAI is a program that can use different AI algorithms to learn how to play Flappy Bird

options:
  -h, --help           show this help message and exit
  -m {DQN,DDQN,CDDQN}  The algorithm you want to train the options are DQN for Deep Q-Network, DDQN for Double Deep Q-Network, CDDQN for Clipped
                       Double Deep Q-Network
  -lr LR               The learning rate of the gradient descent
  -ls LAYER_SIZE       The hidden layer size of the neural networks used by the algorithms
  -g GAMMA             The discount factor of the reward
  -e EPSILON           Epsilon is used in the training of the AI's and describes the probability of taking a random action
  -t TAU               The update factor for the target network
  -n EPISODES          The number of episodes to train
  -s SEED              The seed for the random generators
  -v VIDEO             The path to where the videos should be stored
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
and therefore using the default parameters.

![Comparison](https://github.com/NicolaWidmer/FlappyAI/assets/61154523/cf5f58ff-4e01-4724-afc2-1f47ee9b70ea)


The train/evaluation rewards plotted are averaged over the last 100 episodes otherwise the plot would be much more noisy. The gap between training and evaluation reward happens because during training with probability $\epsilon$ a random action is chosen, which may not be optimal. One can also see the typical problem of classical Deep Q-learning after some time, it gets overconfident about the learned Q-Value and begins to perform worse. Things that Double Deep Q-learning and Clipped Double Deep Q-Learning to by some extent. 

Below are videos of one episode during training for different algorithms and different numbers of train episodes.

### DQN

After 200 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/3d7d2207-cbb6-4bd6-86ee-0374fd5e41ff

After 600 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/c1911b71-16dc-44c7-9be6-b1a3da5b88bb

After 1000 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/d243a261-b763-4610-8263-93b98fdd9c3c

### Double DQN

After 200 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/4a99547c-bd36-447f-a523-8728d5c06609

After 600 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/c7994778-900f-4329-809b-16fa00fb5719

After 1000 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/42b2f8e5-8a83-43db-8858-7f8442411206


### Clipped Double DQN

After 200 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/b188fa00-7c60-4a15-a292-a7934c64b389

After 600 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/8dfd7bb5-5069-487e-bd4c-09853105693a

After 1000 episodes

https://github.com/NicolaWidmer/FlappyAI/assets/61154523/d5fbf1bc-8947-4b7c-a12f-030bd0a94b74
