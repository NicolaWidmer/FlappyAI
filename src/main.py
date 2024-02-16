
from Game import Game
from Agent import Agent
from DeepQ.DQN import DQN
from DeepQ.DoubleDQN import DoubleDQN
from DeepQ.ClippedDoubleDQN import ClippedDoubleDQN
import argparse
import random
import torch
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import pygame

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.comparison:
        compare_models(args)
        return
    
    if args.play:
        game = Game()
        actor = lambda x: 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0
        game.run(actor)
        return
    
    run(args)
        

def run_episode(game:Game,agent:Agent,train:bool,max_rounds:int):

    game.reset()
    state = game.get_state()

    crashed = False
    rounds = 0
    reward_sum = 0

    while not crashed and rounds < max_rounds:
        action = agent.get_action(state,train=train)

        next_state, reward, crashed = game.step(action)

        if train:
            agent.put(state,action,reward,next_state,0 if crashed else 1)
            agent.train_agent()

        state = next_state
        rounds += 1
        reward_sum += reward

    return reward_sum

def make_video(game:Game,agent:Agent,title:str):
    game.make_video(lambda x: agent.get_action(x,train=False),title)

def train_model(game:Game,args):

    if args.model == "DQN":
        agent = DQN(args.layer_size,args.lr,args.gamma,args.epsilon)
    elif args.model == "DDQN":
        agent = DoubleDQN(args.layer_size,args.lr,args.gamma,args.epsilon,args.tau)
    elif args.model == "CDDQN":
        agent = ClippedDoubleDQN(args.layer_size,args.lr,args.gamma,args.epsilon,args.tau)

    name = agent.__class__.__name__
    train_rewards = []
    mean_train_reward = []
    eval_rewards = []
    mean_eval_reward = []

    for i in range(args.episodes):

        train_rewards.append(run_episode(game,agent,True,1000))
        mean_train_reward.append(mean(train_rewards[-100:]))

        eval_rewards.append(run_episode(game,agent,False,1000))
        mean_eval_reward.append(mean(eval_rewards[-100:]))

        if (i+1)%100==0:
            print(f"Completed {i+1} episodes of training using the {name} algorithm.")
            print(f"The average evaluation reward for the last 100 episodes is {mean_eval_reward[-1]}")
            print(f"The average train reward for the last 100 episodes is {mean_train_reward[-1]}")
            print()

        if (i+1)%200==0 and args.video != "None":
            vidoe_path = f"{args.video}/{name}_episode{i+1}"
            game.make_video(lambda x: agent.get_action(x,train=False),vidoe_path)

    return  mean_train_reward,mean_eval_reward


def run(args):

    game = Game()

    mean_train_reward,mean_eval_reward = train_model(game,args)

    name = args.model

    plt.plot(mean_train_reward,label = "train reward")
    plt.plot(mean_eval_reward,label = "evaluation reward")

    plt.title(f"{name} train")
    plt.xlabel("Training episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def compare_models(args):

    models = ["CDDQN","DQN","DDQN"]
    
    mean_train_rewards = []
    mean_eval_rewards = []

    for name in models:
        game = Game()
        args.model = name
        mean_train_reward,mean_eval_reward = train_model(game,args)

        mean_train_rewards.append(mean_train_reward)
        mean_eval_rewards.append(mean_eval_reward)

    for name,mean_train_reward,mean_eval_reward in zip(models,mean_train_rewards,mean_eval_rewards):
        plt.plot(mean_train_reward,label = f"{name} train reward")
        plt.plot(mean_eval_reward,label = f"{name} evaluation reward")


    plt.title("Comparison of different RL algorithms")
    plt.xlabel("Training episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
        

def parse_args():

    agent_dict = {
        "DQN": "Deep Q-Network",
        "DDQN": "Double Deep Q-Network",
        "CDDQN": "Clipped Double Deep Q-Network",
    }

    model_options = ""
    for key,value in agent_dict.items():
        model_options += f"{key} for {value}, "
    model_options = model_options[:-2]

    parser = argparse.ArgumentParser(description= "FlappyAI is a programm which can usedifferent AI algorithms to learn how to play flappy bird")

    parser.add_argument('-m',
                        action="store", dest="model",
                        choices=agent_dict.keys(),
                        help='The alorithm you want to train the options are '+ model_options,
                        default='CDDQN')
    
    parser.add_argument('-lr',
                        action="store", dest="lr",
                        type=float,
                        help='The learning rate of the gradient descent',
                        default='0.001')
    
    parser.add_argument('-ls',
                        action="store", dest="layer_size",
                        type=int,
                        help='The hidden layer size of the neural networks used by the alorithms',
                        default='50')
    
    parser.add_argument('-g',
                        action="store", dest="gamma",
                        type=float,
                        help='The discount factor of the reward',
                        default='0.99')
        
    parser.add_argument('-e',
                        action="store", dest="epsilon",
                        type=float,
                        help='Epsilon is used in the training of the AI\'s and describes the probability of takeing a random action',
                        default='0.05')
     
    parser.add_argument('-t',
                        action="store", dest="tau",
                        type=float,
                        help='The update factor for the target network',
                        default='0.01')
    
    parser.add_argument('-n',
                        action="store", dest="episodes",
                        type=int,
                        help='The number of episodes to train',
                        default='1000')
    
    parser.add_argument('-s',
                        action="store", dest="seed",
                        type=int,
                        help='The seed for the random generators',
                        default='4444')
    
    parser.add_argument('-v',
                        action="store", dest="video",
                        help='The path to where to video of the result should be stored',
                        default='None')
    
    parser.add_argument('-c',
                        action='store_true', dest="comparison",
                        help='Run all algorithms and compare them')    
    
    parser.add_argument('-p',
                        action='store_true', dest="play",
                        help='Play the game yourself')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()