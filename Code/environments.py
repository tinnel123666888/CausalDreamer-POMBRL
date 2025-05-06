# -*- coding=UTF-8 -*-
# __init__.py
from os.path import dirname, join, abspath
import os
import sys
import time
#import cv2
#import pygame
import random
from multiprocessing import Process, Queue
import signal
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from pynput.keyboard import Key,Listener
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import heapq

K_epoch = 1
T_horizon = 20
batch_size = 1
buffer_limit = 1000
batch_rea_size = 1
symbol = 'save'

class ant_maze():

    def __init__(self, map_id, rand_factor):
        super(ant_maze, self).__init__()
        plt.ion()
        self.lock = 0
        self.rand_factor = copy.deepcopy(rand_factor)
        self.map_id = copy.deepcopy(map_id)
        self.r_coll = 0

    def on_press(self, index, flag):
        if self.map[self.agent[0]][self.agent[1]] == 3:
            coin = random.random()
            if coin <= self.rand_factor:
                choice = random.choice([0, 1, 2, 3])
                if choice == 0:
                    self.agent[1] += 1
                if choice == 1:
                    self.agent[1] -= 1
                if choice == 2:
                    self.agent[0] += 1
                if choice == 3:
                    self.agent[0] -= 1
            else:
                if index == 't':
                    self.agent[1] += 1
                    if self.pos_checker():
                        self.agent[1] -= 1

                elif index == 'g':
                    self.agent[1] -= 1
                    if self.pos_checker():
                        self.agent[1] += 1

                elif index == 'f':
                    self.agent[0] += 1
                    if self.pos_checker():
                        self.agent[0] -= 1

                elif index == 'h':
                    self.agent[0] -= 1
                    if self.pos_checker():
                        self.agent[0] += 1
        else:
            if index == 't':
                self.agent[1] += 1
                if self.pos_checker():
                    self.agent[1] -= 1

            elif index == 'g':
                self.agent[1] -= 1
                if self.pos_checker():
                    self.agent[1] += 1

            elif index == 'f':
                self.agent[0] += 1
                if self.pos_checker():
                    self.agent[0] -= 1

            elif index == 'h':
                self.agent[0] -= 1
                if self.pos_checker():
                    self.agent[0] += 1

            else:
                pass

    def pos_checker(self):

        if self.map[self.agent[0]][self.agent[1]] == 4:
            self.r_coll = -0.1
            return(1)
        else:
            self.r_coll = 0
            return(0)

    def reset(self):
        self.done = 0
        self.lock = 0
        self.agent = [9, 7]
        coin = random.random()


        if self.map_id == 1:
            if coin < 0.5:
                self.map = [  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4],
                              [4, 4, 4, 4, 6, 14, 13, 10, 11, 12, 5, 4, 4, 4,4],
                              [4, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4,4],
                              [4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 7, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 3, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 2, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 0, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

                state_info = 0
            else:
                self.map = [  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4],
                              [4, 4, 4, 4, 5, 12, 11, 10, 13, 14, 6, 4, 4, 4,4],
                              [4, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4,4],
                              [4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 7, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 3, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 2, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4,  4, 1, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

                state_info = 1
        if self.map_id == 0:
            if coin < 0.5:
                self.map = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4,  4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4,  4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 11, 5, 4, 6, 11, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 10, 4, 4, 4, 14, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 9,  8, 7, 12, 13, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  3, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  2, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  0, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
                state_info = 0
            else:
                self.map = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4,  4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4,  4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 11, 6, 4, 5, 11, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 14, 4, 4, 4, 10, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 13, 12, 7, 8, 9, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  3, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  2, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  1, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
                state_info = 1
        s = np.zeros(15).tolist()
        s[state_info] = 1
        return s

    def drawer(self):
        plt.clf()
        self.traj_x.append(self.agent[0])
        self.traj_y.append(self.agent[1])
        plt.plot(self.traj_x, self.traj_y)
        plt.xticks(np.arange(0, 10))
        plt.yticks(np.arange(0, 10))
        plt.grid()
        plt.pause(0.1)
        plt.ioff()

    def get_info(self):
        s = copy.deepcopy(self.agent)
        if self.map_id in [0,1,2,5]:
            s[0] = copy.deepcopy(6 - s[0])
            s[1] = copy.deepcopy(s[1])
        r = 0
        done = 0

        if self.map_id in [0,1,2,3]:
            if self.map[self.agent[0]][self.agent[1]] == 5:
                r = 1
                done = 1
            elif self.map[self.agent[0]][self.agent[1]] == 6:
                    r = 0
                    done = 1
            if self.map[self.agent[0]][self.agent[1]] == 7:
                state_info = 7
            else:
                state_info = self.map[self.agent[0]][self.agent[1]]
            s = np.zeros(15).tolist()
            s[state_info] = 1
            return (s, r, done)



