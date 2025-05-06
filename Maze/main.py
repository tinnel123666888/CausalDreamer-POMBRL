# -*- coding:utf-8 -*-
import argparse
import torch


K_epoch = 1
T_horizon = 20
batch_size = 1
buffer_limit = 1000
batch_rea_size = 1
symbol = 'save'
choice_id = 0

def to_tensor(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    return x

def to_LonTensor(x):
    x = torch.LongTensor(x)
    return x

def my_sample(prob_list):
    coin = random.random()
    if coin < prob_list[0]:
        sample_id = 0
    if coin > prob_list[0] and coin < prob_list[1] + prob_list[0]:
        sample_id = 1
    return sample_id


def main_dmc(game_name, level, time_flag, cud):

    import datetime

    # -----normal------------
    import random
    import copy
    import time
    import os
    import sys
    import numpy as np
    from collections import deque
    import itertools as it
    import math
    import datetime
    from multiprocessing import Process, Queue
    import signal
    # -----network-----------
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    from torch.autograd import Variable
    import time
    from multiprocessing import Process, Queue
    import signal
    from environments import ant_maze
    from models import DMC

    cuda_device = torch.device('cuda:%d' % (cud))
    def dmc_maze(game_name, level, time_flag, q1, q2):
        action_list = ['t', 'g', 'f', 'h']
        map_id = level-1
        game = ant_maze(map_id, 0)
        agent = DMC(game_name, map_id, time_flag, cud).cuda(device=cuda_device)
        episode_num = 0
        sumt = 0
        while (1):
            episode_num += 1
            s = game.reset()
            
            if time_flag in [0, 7]:
                    h_i = torch.autograd.Variable(to_tensor(np.zeros((1,100)))).cuda(device=cuda_device)
                    c_i = torch.autograd.Variable(to_tensor(np.zeros((1,100)))).cuda(device=cuda_device)
            if time_flag == 1:
                    agent.clean_hci_buff()
            if time_flag in [2, 3, 4, 5]:
                    h_i = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=cuda_device)
                    c_i = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=cuda_device)
            if time_flag == 6:
                    _map = torch.autograd.Variable(to_tensor(np.zeros((1, 20)))).cuda(device=cuda_device)
            agent.clean_hci_buff()
            tran_1 = []
            for t in range(30):
                if 1:
                    if time_flag == 0:
                        s_moved = torch.autograd.Variable(torch.from_numpy(np.array([[s]])).float()).cuda(device=cuda_device)
                        a, h_i, c_i = agent.choose_action_lstm(s_moved, 0.3, h_i, c_i)
                    if time_flag == 1:
                        a = random.choice(range(4))
                    if time_flag == 2:
                        a, fl = agent.choose_action_hrl([[s]], 0.3)
                    if time_flag == 3:
                        a, fl, h_i, c_i = agent.choose_action_hci([[s]], 0.3, h_i, c_i)
                    if time_flag == 4:
                        a = agent.choose_action_attention([[s]], 0.3)
                    if time_flag == 5:
                        a = agent.choose_action_former([[s]], 0.3)
                    if time_flag == 6:
                        s_moved = torch.autograd.Variable(torch.from_numpy(np.array([[s]])).float()).cuda(device=cuda_device)
                        a, _map = agent.choose_action_map(s_moved, 0.3, _map)
                    if time_flag == 7:
                        s_moved = torch.autograd.Variable(torch.from_numpy(np.array([[s]])).float()).cuda(device=cuda_device)
                        a, h_i = agent.choose_action_gru(s_moved, 0.3, h_i)

                if episode_num > 100000:
                    print(a, fl)
                game.on_press(action_list[a], 0)
                (s_prime, r, done) = game.get_info()
                #sum_reward += r
                s_p = copy.deepcopy(s_prime)
                
                if time_flag in [0, 7]:
                        tran = [s, a, r / 1.0, s_p, done, h_i.data.cpu().numpy()[0], c_i.data.cpu().numpy()[0]]
                if time_flag in [1, 4, 5, 6, 8]:
                        tran = [s, a, r / 1.0, s_p, done]
                if time_flag in [2, 3]:
                        tran = [s, a, r / 1.0, s_p, done, fl]
                if time_flag == 10:
                        print(s)
                tran_1.append(copy.deepcopy(tran))
                if done == 1:
                    break
                s = copy.deepcopy(s_p)
            if time_flag == 1:
                 if episode_num < 100:
                     if t < 9:
                         sumt += 1
                     print(np.array(tran_1), sumt, episode_num)
                     continue
            q2.put(tran_1)
            while (1):
                if not q1.empty():
                    a = q1.get()

                    if a == 'q':
                        parameters = q1.get()
                        agent.load_state_dict(parameters)
                        gatebuff = q1.get()
                        agent.gatebuff = copy.deepcopy(gatebuff)
                        break
            s = game.reset()
            
            if time_flag in [0, 7]:
                    h_i = torch.autograd.Variable(to_tensor(np.zeros((1,100)))).cuda(device=cuda_device)
                    c_i = torch.autograd.Variable(to_tensor(np.zeros((1,100)))).cuda(device=cuda_device)
            if time_flag == 1:
                    agent.clean_hci_buff()
            if time_flag in [2, 3, 4, 5]:
                    h_i = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=cuda_device)
                    c_i = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=cuda_device)
            if time_flag == 6:
                    _map = torch.autograd.Variable(to_tensor(np.zeros((1,20)))).cuda(device=cuda_device)
            sum_reward = 0
            agent.clean_hci_buff()
            for t in range(30):
                if time_flag == 0:
                    
                    s_moved = torch.autograd.Variable(torch.from_numpy(np.array([s])).float()).cuda(device=cuda_device)
                    a, h_i, c_i = agent.choose_action_lstm(s_moved, 0, h_i, c_i)
                if time_flag == 2:
                    a, fl = agent.choose_action_hrl([[s]], 0)
                if time_flag == 3:
                    a, fl, h_i, c_i = agent.choose_action_hci([[s]], 0, h_i, c_i)
                if time_flag == 4:
                    a = agent.choose_action_attention([[s]], 0)
                if time_flag == 5:
                    a = agent.choose_action_former([[s]], 0)
                if time_flag == 6:
                    s_moved = torch.autograd.Variable(torch.from_numpy(np.array([[s]])).float()).cuda(device=cuda_device)
                    a, _map = agent.choose_action_map(s_moved, 0, _map)
                if time_flag == 7:
                    s_moved = torch.autograd.Variable(torch.from_numpy(np.array([[s]])).float()).cuda(device=cuda_device)
                    a, h_i = agent.choose_action_gru(s_moved, 0, h_i)
                game.on_press(action_list[a], 0)
                (s_prime, r, done) = game.get_info()
                sum_reward += r
                s_p = copy.deepcopy(s_prime)

                if done == 1:
                    break
                s = copy.deepcopy(s_p)
            if 1:
                print('episode_num:', episode_num, 'sum_reward:', sum_reward, 'step:', t)


    p = []
    queues_1 = []
    queues_2 = []

    queues_1.append(Queue())
    queues_2.append(Queue())

    if game_name == 'Maze':
            p.append(Process(target=dmc_maze, args=(game_name, level, time_flag, queues_1[0], queues_2[0])))

    for j in range(len(p)):
        p[j].start()
    map_id = level - 1
    model = DMC(game_name, map_id, time_flag, cud).cuda(device=cuda_device)

    for n_epi in range(500000):
        done_lst = []
        for i in range(len(p)):
            done_lst.append(False)
        while (1):
            for j in range(len(p)):
                if not done_lst[j]:
                    if not queues_2[j].empty():
                        tran_lst = copy.deepcopy(queues_2[j].get())
                        for i in range(len(tran_lst)):
                            model.put_data(tran_lst[i], j)
                        done_lst[j] = True
            done_sym = 1
            for j in range(len(done_lst)):
                done_sym = done_sym and done_lst[j]

            if (done_sym):
                if 1:
                    if time_flag in [0, 7]:
                        model.train_net_gatedrnn()
                    if time_flag == 3:
                        model.train_pi_hci()
                    if time_flag == 4:
                        model.train_sai()
                    if time_flag == 5:
                        model.train_wmg()
                    if time_flag == 6:
                        model.train_map()
                parameters = model.state_dict()
                for u in range(len(p)):
                    queues_1[u].put('q')
                    queues_1[u].put(parameters)
                    queues_1[u].put(model.gatebuff)
                break

def allocate(game_name='Maze', level=1, method='t-hci', cud=0):
    time_flag = ['lstm', '', '', 't-hci', 'sai', 'wmg', 'nm', 'gru', 'gate'].index(method)
    if game_name == 'babyAI':
            main_ac(game_name, level, time_flag, cud)
    if game_name == 'Maze':
            main_dmc(game_name, level, time_flag, cud)


parser = argparse.ArgumentParser()
parser.add_argument('-level', type=str, help='input the number of level')
parser.add_argument('-cud', type=str, help='input the cuda id')
args = parser.parse_args()

multi_ind = 0
allocate(level=int(args.level), cud=int(args.cud))

                            
