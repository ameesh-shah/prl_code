import sys
import itertools
import random
import time
import copy
import getopt
import os

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

NOISE = True
NOISE_FUNC = lambda : np.random.randn(8)*0.2

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
        if collect_data:
            states.append(obs)
            actions.append(action)
        obs, rew, done, info = env_object.step(action)

        collected_reward += rew
        if not collect_data:
            env_object.render()
    if info['finished']:
        return states, actions
    else:
        return None, None
    
def collect_data_for_imitation_learning(env_object, program_method, num_runs=100, outfilepath=None):
    all_state_data = []
    all_action_data = []
    all_paths = []
    for _ in tqdm(range(num_runs)):
        states, actions = deploy_behavior_program_in_maze(env_object, program_method, collect_data=True)
        if states is None:
            continue
        all_paths.append([state[:2] for state in states])
        all_state_data.extend(states)
        all_action_data.extend(actions)
    print("Number of states collected is {}".format(len(all_state_data)))
    if outfilepath is None:
        outfilepath = "IL_base"
    statefilepath = outfilepath + "_states"
    actionfilepath = outfilepath + "_actions"
    pathfilepath = outfilepath + "_paths"
    with open(statefilepath, "wb") as stateout:
        np.save(stateout, all_state_data)
    with open(actionfilepath, "wb") as actionout:
        np.save(actionout, all_action_data)
    with open(pathfilepath, "wb") as pathout:
        np.save(pathout, all_paths)
    print("Data saved to {}, {}, and {}".format(statefilepath, actionfilepath, pathfilepath))


def collect_data_for_generative_learning(env_object, distr=(.5, .25, .25), num_runs=1000, outfilepath=None):
    all_state_data = []
    all_action_data = []
    counts = [0, 0, 0]
    total_num_states = 0
    behaviorprog = AntBehaviorProgram()
    for _ in tqdm(range(num_runs)):
        idx_choice = np.random.choice(range(3), p=distr)
        if idx_choice == 0:
            program_method = behaviorprog.move_ant_to_top_goal
        elif idx_choice == 1:
            program_method = behaviorprog.move_ant_to_left_goal
        else:
            program_method = behaviorprog.move_ant_to_right_goal
        counts[idx_choice] += 1
        states, actions = deploy_behavior_program_in_maze(
            env_object, program_method, collect_data=True)
        all_state_data.append(states)
        all_action_data.append(actions)
        total_num_states += len(states)
    print("Number of trajectories collected is {}".format(len(all_state_data)))
    print("Average number of states per traj is {}".format(total_num_states / num_runs))
    print("Distribution over programs is {}".format(counts))
    if outfilepath is None:
        outfilepath = "generative_base"
    statefilepath = outfilepath + "_states"
    actionfilepath = outfilepath + "_actions"
    with open(statefilepath, "wb") as stateout:
        np.save(stateout, all_state_data)
    with open(actionfilepath, "wb") as actionout:
        np.save(actionout, all_action_data)
    print("Data saved to {} and {}".format(statefilepath, actionfilepath))


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

        return self.up_model.act(formatted_observation, deterministic=True) + NOISE*NOISE_FUNC()
    
    def move_ant_to_left_goal(self, current_observation):
        xypos = get_ant_position(current_observation)
        xpos, ypos = xypos[0], xypos[1]
        current_observation = torch.as_tensor(current_observation, dtype=torch.float32)
        formatted_observation = current_observation[self.index_action_space]
        if xpos < 4.0:
            #print(f"up-act {self.up_model.act(formatted_observation, deterministic=True)}")
            return self.up_model.act(formatted_observation, deterministic=True) + NOISE*NOISE_FUNC()
        else:
            #print(f"left-act {self.left_model.act(formatted_observation, deterministic=True)}")
            return self.left_model.act(formatted_observation, deterministic=True) + NOISE*NOISE_FUNC()
    
    def move_ant_to_right_goal(self, current_observation):
        xypos = get_ant_position(current_observation)
        xpos, ypos = xypos[0], xypos[1]
        current_observation = torch.as_tensor(
            current_observation, dtype=torch.float32)
        formatted_observation = current_observation[self.index_action_space]
        if xpos < 4.0:
            return self.up_model.act(formatted_observation, deterministic=True) + NOISE*NOISE_FUNC()
        else:
            return self.right_model.act(formatted_observation, deterministic=True) + NOISE*NOISE_FUNC()
        
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

# get our env object
e = GymEnv('mjrl_all_goals_ant-v1', expose_all_qpos=False)
ANT_FUNCTIONS = [(f0_ant, 4), (f1_ant, 1), (f2_ant, 1)]
ALL_ANT_FUNCTIONS = [([1, 1, 1], 6)]

exp_name = 'cross'
train_iter = 50
num_traj = 50
tune_iter = 250
arch_iter = 1
prog_iter = 10
input_dict = dict(models=ANT_MODELS, functions=ANT_FUNCTIONS, all_functions=ALL_ANT_FUNCTIONS, input_dim=115, num_action_space=8, index_action_space=range(2, 113))

#deploy_behavior_program_in_maze(e, behaviorprog.move_ant_to_left_goal)
behaviorprog = AntBehaviorProgram()
collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_left_goal, 100, outfilepath=f"../data/ant_maze{'_noise' if NOISE else ''}_left_train")
collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_left_goal, 50, outfilepath=f"../data/ant_maze{'_noise' if NOISE else ''}_left_test")
collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_right_goal, 100, outfilepath=f"../data/ant_maze{'_noise' if NOISE else ''}_right_train")
collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_right_goal, 50, outfilepath=f"../data/ant_maze{'_noise' if NOISE else ''}_right_test")
collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_top_goal, 100, outfilepath=f"../data/ant_maze{'_noise' if NOISE else ''}_top_train")
collect_data_for_imitation_learning(e, behaviorprog.move_ant_to_top_goal, 50, outfilepath=f"../data/ant_maze{'_noise' if NOISE else ''}_top_test")
