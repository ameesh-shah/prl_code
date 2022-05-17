import sys
import itertools
import random
import time
import copy
import getopt
import os
from train import run_train

os.environ['OMP_NUM_THREADS'] = '1'

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dill import pickle
from mjrl.utils.gym_env import GymEnv
import mjrl.envs
import time as timer

ANT_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant/' + direction + '.pt'
    ANT_MODELS.append(torch.load(filename))

def deploy_behavior_program_in_maze(env_object, program_method, collect_data=False):
    # given a programmatic model of behavior, deploy it in a given environment
    #env_object.render()
    states = []
    actions = []
    obs = env_object.reset()
    done = False
    collected_reward = 0
    while not done:
        action = program_method(obs)
        # obs_in = torch.tensor(obs).float().unsqueeze(0)
        # action = program_method(obs_in)
        # action = action.squeeze().detach().numpy()
        if collect_data:
            states.append(obs)
            actions.append(action)
        obs, rew, done, info = env_object.step(action)
        collected_reward += rew
        if not collect_data:
            env_object.render()
    return states, actions
    
def collect_data_for_imitation_learning(env_object, program_method, num_runs=100, outfilepath=None):
    all_state_data = []
    all_action_data = []
    for _ in tqdm(range(num_runs)):
        states, actions = deploy_behavior_program_in_maze(env_object, program_method, collect_data=True)
        all_state_data.extend(states)
        all_action_data.extend(actions)
    print("Number of states collected is {}".format(len(all_state_data)))
    if outfilepath is None:
        outfilepath = "IL_base"
    statefilepath = outfilepath + "_states"
    actionfilepath = outfilepath + "_actions"
    with open(statefilepath, "wb") as stateout:
        np.save(stateout, all_state_data)
    with open(actionfilepath, "wb") as actionout:
        np.save(actionout, all_action_data)
    print("Data saved to {} and {}".format(statefilepath, actionfilepath))

def collect_data_for_gail(env_object, num_runs=100):
    all_state_data = []
    all_action_data = []
    for _ in tqdm(range(num_runs)):
        traj_choice = np.random.choice(3, p=[0.5, 0.25, 0.25])
        behaviorprog = AntBehaviorProgram()
        if traj_choice == 0:
            program_method = behaviorprog.move_ant_to_top_goal
        elif traj_choice == 1:
            program_method = behaviorprog.move_ant_to_left_goal
        else:
            program_method = behaviorprog.move_ant_to_right_goal
        states, actions = deploy_behavior_program_in_maze(env_object, program_method, collect_data=True)
        all_state_data.extend(states)
        all_action_data.extend(actions)
    print("Number of states collected is {}".format(len(all_state_data)))
    state_action_tuples = []
    for i in range(len(all_state_data)):
        state_action_tuples.append((all_state_data[i], all_action_data[i]))
    return state_action_tuples


class AntBehaviorProgram:

    def __init__(self) -> None:
        self.counter = 0
        self.index_action_space = range(2, 113)
        self.ANT_MODELS = []
        for direction in ['up', 'down', 'left', 'right']:
            filename = os.getcwd() + '/primitives/ant/' + direction + '.pt'
            self.ANT_MODELS.append(torch.load(filename))
        self.up_model = self.ANT_MODELS[0]
        self.down_model = self.ANT_MODELS[1]
        self.left_model = self.ANT_MODELS[2]
        self.right_model = self.ANT_MODELS[3]
    
    def reset_program_info(self):
        self.counter = 0

    def move_ant_to_top_goal(self, current_observation):
        #move the ant forward deterministically
        current_observation = torch.as_tensor(current_observation, dtype=torch.float32)
        formatted_observation = current_observation[self.index_action_space]
        self.counter += 1

        return self.up_model.act(formatted_observation, deterministic=True)
    
    def move_ant_to_left_goal(self, current_observation):
        xypos = get_ant_position(current_observation)
        xpos, ypos = xypos[0], xypos[1]
        current_observation = torch.as_tensor(current_observation, dtype=torch.float32)
        formatted_observation = current_observation[self.index_action_space]
        if xpos < 4.0:
            return self.up_model.act(formatted_observation, deterministic=True)
        else:
            return self.left_model.act(formatted_observation, deterministic=True)
    
    def move_ant_to_right_goal(self, current_observation):
        xypos = get_ant_position(current_observation)
        xpos, ypos = xypos[0], xypos[1]
        current_observation = torch.as_tensor(
            current_observation, dtype=torch.float32)
        formatted_observation = current_observation[self.index_action_space]
        if xpos < 4.0:
            return self.up_model.act(formatted_observation, deterministic=True)
        else:
            return self.right_model.act(formatted_observation, deterministic=True)
        
def get_ant_position(full_observation):
    return full_observation[0:2]

def f0_ant(x):
    # return XY + GXY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    GX = Variable(x[:, -2:-1], requires_grad=False)
    GY = Variable(x[:, -1:], requires_grad=False)
    return Variable(torch.cat((X, Y, GX, GY), dim=1), requires_grad=False)


def f1_ant(x):
    # return THETA_XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.atan2(Y, X), requires_grad=False)


def f2_ant(x):
    # return DISTANCE_XY
    X = Variable(x[:, 0:1], requires_grad=False)
    Y = Variable(x[:, 1:2], requires_grad=False)
    return Variable(torch.linalg.norm(torch.cat((X, Y), dim=1), dim=1, keepdim=True), requires_grad=False)

def get_gym_env():
    return GymEnv('mjrl_cross_maze_ant_random-v1', expose_all_qpos=False)

# get our env object
e = GymEnv('mjrl_cross_maze_ant_random-v1', expose_all_qpos=False)
ANT_FUNCTIONS = [(f0_ant, 4), (f1_ant, 1), (f2_ant, 1)]
ALL_ANT_FUNCTIONS = [([1, 1, 1], 6)]

exp_name = 'cross'
train_iter = 50
num_traj = 50
tune_iter = 250
arch_iter = 1
prog_iter = 10
input_dict = dict(models=ANT_MODELS, functions=ANT_FUNCTIONS, all_functions=ALL_ANT_FUNCTIONS, input_dim=115, num_action_space=8, index_action_space=range(2, 113))

#behaviorprog = AntBehaviorProgram()
#deploy_behavior_program_in_maze(e, behaviorprog.move_ant_to_left_goal)
#collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_left_goal, 10, outfilepath="ant_maze_left_test")

#model = run_train("ant_maze_left_test")
#deploy_behavior_program_in_maze(e, model)