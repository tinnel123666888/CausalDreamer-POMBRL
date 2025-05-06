# -*- coding=UTF-8 -*-
# __init__.py

import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy
import numpy as np
import time

cuda_device = torch.device('cuda:0')
cuda_device2 = torch.device('cuda:1')
cuda_device3 = torch.device('cuda:2')
cuda_device4 = torch.device('cuda:3')

z, c = [], []

def to_tensor(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    return x

def to_LonTensor(x):
    x = torch.LongTensor(x)
    return x

def calculate_reward(tran):
    symbol_lst = np.load('savings/symbol_list.npy')
    tran = copy.deepcopy(tran)
    proit_s, proit_sp, done = tran
    block = 0

    if proit_sp.tolist() in symbol_lst.tolist():
        block = 1

    if proit_s.tolist() not in symbol_lst.tolist() and proit_sp.tolist() in symbol_lst.tolist() and not done:
        reward = 1
    else:
        reward = 0

    return reward, block


class gate_f(nn.Module):

    def __init__(self):
            super(gate_f, self).__init__()
            self.fc1 = nn.Linear(1, 400)
            self.fc2 = nn.GRUCell(400, 400)
            self.fc3 = nn.Linear(400, 100)
            self.fc_G = nn.Linear(100, 2)

class inverse_pre(nn.Module):
    def __init__(self):
            super(inverse_pre, self).__init__()
            self.fc1 = nn.Linear(15, 400)
            self.fc2 = nn.Linear(400, 400)
            self.gru = nn.GRUCell(400, 800)
            self.fc3 = nn.Linear(800, 200)
            self.out = nn.Linear(200, 2)

class DMC(nn.Module):

    def __init__(self, game_name, map_id, time_flag, cuda_index):
      super(DMC, self).__init__()
      self.lstm_factor = 1
      self.gamma = 0.95
      self.game_name = copy.deepcopy(game_name)
      self.learning_rate = 0.0005
      self.data = []
      self.data2 = []
      self.lock = 0
      self.zeros_length = 22
      self.add_vec = [1]*15
      self.add_label = 0
      self.time_flag = copy.deepcopy(time_flag)
      self.cuda_device = torch.device('cuda:%d'%(cuda_index))
      self.offdata = []
      self.observations = []
      self.epi_num = 0
      self.map_id = copy.deepcopy(map_id)
      self.action = [0, 1, 2, 3]
      processes_num = 1
      self.proc_num = 1
      if self.time_flag in [1, 3]:
          for i in range(processes_num):
              self.data.append([])
      else:
          for i in range(processes_num):
              self.data.append([])

      if self.game_name == 'Maze':
        self.action = [0, 1, 2, 3]
        self.gatebuff = []
        if self.time_flag == 0:
            self.fc1 = nn.Linear(15, 100)
            self.fc2 = nn.Linear(100, 100)
            self.lstm = nn.LSTMCell(100, 100)
            self.fc3 = nn.Linear(100, 100)
            self.fc_Q = nn.Linear(100, 4)
        if self.time_flag in [1, 2]:
            self.fc1_do = nn.Linear(1, 50)
            self.fc1_hrl1 = nn.Linear(1,50)
            self.fc1_hrl2 = nn.Linear(1,100)

            self.lstm12 = nn.LSTMCell(100, 100)
            self.fc_Q12 = nn.Linear(100, 4)
            self.fc_pr = nn.Linear(100,4)

            self.fc11 = nn.Linear(50, 50)
            self.fc_Q11 = nn.Linear(50, 4)

            self.fc_do1 = nn.Linear(50, 50)
            self.fc_do2 = nn.Linear(50, 2)
            
            self.attention_fcdo1 = nn.Linear(1, 100)
            self.attention_fcdo2 = nn.Linear(100, 100)
            self.attention_do = nn.Linear(100, 2)
            

            self.attention_fcvalue1 = nn.Linear(1, 100)
            self.attention_fcvalue2 = nn.Linear(100, 100)
            self.attention_value = nn.Linear(100, 20)
            self.attention_wta = nn.Linear(100, 1)

            self.wta_pr_fc1 = nn.Linear(1, 100)
            self.wta_lstm = nn.LSTMCell(100, 100)
            self.attention_pr_fc1 = nn.Linear(40, 100)
            self.attention_pr_fc2 = nn.Linear(100, 100)
            self.attention_pr_out = nn.Linear(100, 4)

        if self.time_flag == 3:
            self.fc1 = nn.Linear(15, 100)
            self.fc2 = nn.Linear(100, 100)
            self.lstm = nn.GRUCell(100, 100)
            self.fc3 = nn.Linear(100, 100)
            self.fc_Q = nn.Linear(100, 4)
            self.inv_pre = inverse_pre()
            self.optimizer_inv = optim.Adam(self.inv_pre.parameters(), lr=self.learning_rate*2)

        if map_id in [0,1,2,3]:
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward_gate(self, x):

        if self.game_name == 'Maze':
            x = F.relu(self.gate_f.fc1(x))
            x = F.relu(self.gate_f.fc2(x))
            x = F.relu(self.gate_f.fc3(x))
            gate = self.gate_f.fc_G(x)

        return gate


    def forward_lstm(self, x, h, c):
        if self.game_name == 'Maze':
            if self.time_flag in [0,7]:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                h, c = self.lstm(x, (h, c))
                x = F.relu(self.fc3(h))
                q = self.fc_Q(x)
            if self.time_flag in [1,3]:
                x_view = x.view(-1,15)
                x = F.relu(self.fc1(x_view))
                x = F.relu(self.fc2(x))
                h = self.lstm(x, h)
                x = self.fc3(h)
                q = self.fc_Q(x)
        return q, h, c

    def clean_hci_buff(self):
        self.h_buff = []
        self.memo_length = 0

    def put_data(self, transition, data_ind):
        self.data[data_ind].append(copy.deepcopy(transition))

    def Gate(self, x):
        if self.game_name == 'Maze':
            if x in self.gatebuff:
                return 0
            else:
                self.memo_length += 1
                return 1

        else:
            state_moved = torch.autograd.Variable(torch.from_numpy(np.array([[x]])).float()).cuda(
                device=self.cuda_device)
            out = self.forward_gate(state_moved).data.cpu().numpy()
            return np.argmax(out[0])

    def choose_action_hci(self, x, epsilon, h, c):
        if self.game_name == 'Maze':
            fl = self.Gate(x[0][0].index(1))
        state_moved = torch.autograd.Variable(torch.from_numpy(np.array(x)).float()).cuda(
            device=self.cuda_device)
        if fl == 1:
            out_hrl, h, c = self.forward_lstm(state_moved, h, c)
        else:
            out_hrl, _, _ = self.forward_lstm(state_moved, h, c)
        coin = random.random()
        if coin < epsilon:
            a = random.choice(self.action)
        else:
            a = out_hrl.data.cpu().numpy()[0].argmax()
        return a, fl, h, c

    def forward_pr(self, x, h, c):
        x = F.relu(self.inv_pre.fc1(x))
        x = self.inv_pre.fc2(x)
        h = self.inv_pre.gru(x, h)
        x = self.inv_pre.fc3(h)
        out_pr = self.inv_pre.out(x)
        return out_pr, h, c

    def choose_action_lstm(self, obs, epsilon, h, c):
        if self.game_name in ['babyAI', 'DIY_Jaco']:
            if self.time_flag == 0:
                out, h, c = self.forward_lstm(obs, h, c)
                coin = random.random()
                if coin < epsilon:
                    return random.choice(self.action), h, c

                else:
                    return out.data.cpu().numpy()[0].argmax(), h, c
          
        else:        
            if self.time_flag == 0:
                out, h, c = self.forward_lstm(obs, h, c)
                coin = random.random()
                if coin < epsilon:
                    return random.choice(self.action), h, c
                else:
                    return out.data.cpu().numpy().argmax(), h, c
            if self.time_flag == 7:
                out, h = self.forward_gru(obs, h)
                coin = random.random()
                if coin < epsilon:
                    return random.choice(self.action), h, c
                else:
                    return out.data.cpu().numpy().argmax(), h, c
            if self.time_flag == 1:
                h = torch.autograd.Variable(to_tensor(np.zeros((1, 50)))).cuda(device=self.cuda_device)
                c = torch.autograd.Variable(to_tensor(np.zeros((1, 50)))).cuda(device=self.cuda_device)
                for i in range(len(self.h_buff)):
                    CI_compose = torch.autograd.Variable(torch.from_numpy(np.array([[obs, self.h_buff[i]]])).float()).cuda(device=self.cuda_device)
                    Do_out = self.forward_do(CI_compose)
                    if Do_out.data.cpu().numpy()[0].argmax() == 0:
                        out, h, c = self.forward_lstm(torch.autograd.Variable(torch.from_numpy(np.array([[self.h_buff[i]]])).float())
                                                  .cuda(device=self.cuda_device), h, c)
                out, h, c = self.forward_lstm(torch.autograd.Variable(torch.from_numpy(np.array([[obs]])).float()).cuda(device=self.cuda_device), h, c)

                coin = random.random()
                if coin < epsilon:
                    return random.choice(self.action)

                else:
                    return out.data.cpu().numpy().argmax()

    def make_batch_lstm(self):
        mini_batches = []
        for i in range(len(self.data)):
            mini_batches.append(copy.deepcopy(self.data[i]))
        batch_lst = []
        for mini_batch in mini_batches:
                batch = []
                if self.game_name == 'Maze':
                    if self.time_flag in [0, 7]:
                        s_lst, a_lst, r_lst, s_prime_lst, done_lst, v_tar, h_lst, c_lst = [], [], [], [], [], [], [], []
                        s_off_lst = []
                        for transition in mini_batch:
                            s, a, r, s_prime, done, h, c = transition
                            s_lst.append([s])
                            s_off_lst.append(s)
                            a_lst.append(a)
                            r_lst.append(r)
                            h_lst.append(h)
                            c_lst.append(c)
                            s_prime_lst.append(s_prime)
                            done_mask = 0 if done else 1
                            done_lst.append(done_mask)
                        for i in range(len(r_lst)):
                            if i == 0:
                                v_tar.append(r_lst[len(r_lst)-1-i])
                            else:
                                v_tar.append(r_lst[len(r_lst)-1-i] + self.gamma * v_tar[-1])
                        #offbatch = [s_off_lst, a_lst, s_prime_lst, h_lst, c_lst, r_lst]
                        v_tar.reverse()
                        v_tar = np.array(v_tar)
                        s_lst = np.array(s_lst)
                        a_lst = np.array(a_lst)
                        batch = [s_lst, a_lst, v_tar, h_lst, c_lst]

                    if self.time_flag in [1, 2]:
                        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
                        for g1 in range(len(mini_batch)):
                            
                            s, a, r, s_prime, done = mini_batch[g1]
                            if s_prime != s and s not in s_lst:
                                s_lst.append(s)
                                a_lst.append(a)
                                r_lst.append(r)
                                s_prime_lst.append(s_prime)

                        s_lst = np.array(s_lst)
                        a_lst = np.array(a_lst)

                        batch = [s_lst, s_prime_lst, a_lst, r_lst]
                    
                    if self.time_flag == 3:
                        s_lst, a_lst, r_lst, s_prime_lst, done_lst, v_tar, fl_lst = [], [], [], [], [], [], []
                        for transition in mini_batch:
                            s, a, r, s_prime, done, fl = transition
                            s_lst.append(s)
                            a_lst.append(a)
                            r_lst.append(r)
                            fl_lst.append(fl)
                            s_prime_lst.append(s_prime)
                            done_mask = 0 if done else 1
                            done_lst.append(done_mask)
                        for i in range(len(r_lst)):
                            if i == 0:
                                v_tar.append(r_lst[len(r_lst)-1-i])
                            else:
                                v_tar.append(r_lst[len(r_lst)-1-i] + self.gamma * v_tar[-1])

                        v_tar.reverse()
                        v_tar = np.array(v_tar)
                        s_lst = s_lst
                        a_lst = a_lst
                        #offbatch = [s_lst, a_lst, s_prime_lst, 0, 0, r_lst]
                        batch = [s_lst, a_lst, v_tar, fl_lst, s_prime_lst]
                        self.epi_num += 1
                        if len(self.offdata) == 0:
                            self.set_offdata([s_lst, a_lst, s_prime_lst, r_lst])
                        elif self.epi_num <= 2002:
                            self.set_offdata([s_lst, a_lst, s_prime_lst, r_lst])
                    batch_lst.append(batch)
                    #self.offdata.append(offbatch)

        self.data = []
        for i in range(self.proc_num):
             self.data.append([])
        return batch_lst

    def set_offdata(self, batch):
        s_lst, a_lst, sp_lst, r_lst = batch
        if len(self.offdata) == 0:
            self.offdata = [[[]], [[]], [[]], [[]]]
            for ind in range(self.zeros_length):
                self.offdata[0][-1].append(self.add_vec)
                self.offdata[1][-1].append(1)
                self.offdata[2][-1].append(self.add_vec)
                self.offdata[3][-1].append([0,1])
        else:
            # time1 = time.time()
            length_off = len(self.offdata[0])
            if len(s_lst) <= self.zeros_length - 2:
                for ind in range(len(s_lst)):
                    if s_lst[ind] not in self.observations:
                        self.observations.append(s_lst[ind])
                    # if sp_lst[ind] not in self.observations:
                    #     self.observations.append(sp_lst[ind])
                    if s_lst[ind] != sp_lst[ind]:
                        if len(self.offdata[0]) == length_off:
                            for i in range(4):
                                self.offdata[i].append([])
                        self.offdata[0][-1].append(s_lst[ind])
                        self.offdata[1][-1].append(a_lst[ind])
                        self.offdata[2][-1].append(sp_lst[ind])
                        r_label = [0, 0]
                        r_label[int(r_lst[ind])] = 1
                        self.offdata[3][-1].append(r_label)

            #     time2 = time.time()
                while len(self.offdata[0][-1]) < self.zeros_length:
                    self.offdata[0][-1].append(self.add_vec)
                    self.offdata[1][-1].append(1)
                    self.offdata[2][-1].append(self.add_vec)
                    self.offdata[3][-1].append([0,1])
                # time3 = time.time()
            #print(self.offdata[0][-1], self.offdata[0][0:-1])
            #print('0:', len(self.offdata[0]))
                if len(self.offdata[0]) >= 2:
                    if self.offdata[0][-1] in self.offdata[0][0:-1]:
                        for i in range(4):
                            self.offdata[i]=self.offdata[i][0:-1]

    def train_net_lstm(self):
            batch_lst = self.make_batch_lstm()
            loss = 0

            for batch in batch_lst:

                h = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=self.cuda_device)
                c = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=self.cuda_device)
                s_lst, a_lst, v_tar_lst, h_buff, c_buff = copy.deepcopy(batch)
                h_lst = []
                c_lst = []
                h_lstp = []
                c_lstp = []
                h_lst.append(h.data.cpu().numpy()[0])
                c_lst.append(c.data.cpu().numpy()[0])

                for i in range(len(s_lst)):
                    h = copy.deepcopy(h_buff[i])
                    c = copy.deepcopy(c_buff[i])
                    if i < len(s_lst) - 1:
                        h_lst.append(h)
                        c_lst.append(c)
                    h_lstp.append(h)
                    c_lstp.append(c)

                if self.game_name in ['Maze', 'babyAI']:
                    state_moved = torch.autograd.Variable(torch.from_numpy(np.array(s_lst)).float()).cuda(
                        device=self.cuda_device)
                h_lst = torch.autograd.Variable(torch.from_numpy(np.array(h_lst)).float()).cuda(
                    device=self.cuda_device)
                c_lst = torch.autograd.Variable(torch.from_numpy(np.array(c_lst)).float()).cuda(
                    device=self.cuda_device)

                if self.time_flag == 7:
                    Q_st, _ = self.forward_gru(state_moved, h_lst)
                else:
                    Q_st, _, _ = self.forward_lstm(state_moved, h_lst, c_lst)
                Q_tar = Q_st.data.cpu().numpy()
                for i in range(len(Q_tar)):
                    Q_tar[i][a_lst[i]] = v_tar_lst[i]
                Q_tar = torch.autograd.Variable(torch.from_numpy(Q_tar).float()).cuda(device=self.cuda_device)
                loss += F.smooth_l1_loss(Q_st, Q_tar.detach())

            loss = loss/len(batch_lst)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_pi_hci(self):

        if self.game_name == 'Maze':
            batch_lst = self.make_batch_lstm()
            for batch in batch_lst:
                s, a, v_tar, fl, sp = batch
                h = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=self.cuda_device)
                c = torch.autograd.Variable(to_tensor(np.zeros((1, 100)))).cuda(device=self.cuda_device)
                for u in range(len(s)):
                    state_moved = torch.autograd.Variable(torch.from_numpy(np.array([[s[u]]])).float()).cuda(
                        device=self.cuda_device)

                    if fl[u] == 1:
                        if u == 0:
                            out_hci = self.forward_lstm(state_moved, h, c)[0][0][a[u]].unsqueeze(0)
                        else:
                            out_hci = torch.cat((out_hci, self.forward_lstm(state_moved, h, c)[0][0][a[u]].unsqueeze(0)), 0)
                        _, h, c = self.forward_lstm(state_moved, h, c)
                    else:
                        if u == 0:
                            out_hci = self.forward_lstm(state_moved, h, c)[0][0][a[u]].unsqueeze(0)
                        else:
                            out_hci = torch.cat((out_hci, self.forward_lstm(state_moved, h, c)[0][0][a[u]].unsqueeze(0)), 0)
            v_tar_moved = torch.autograd.Variable(torch.from_numpy(np.array(v_tar)).float()).cuda(device=self.cuda_device)
            loss = F.smooth_l1_loss(out_hci, v_tar_moved)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.epi_num % 504 == 0 and len(self.offdata[0])>1 and self.epi_num<=2100:#len(self.offdata[0])%3104==0
                self.train_ctf_do2()

    def train_ctf_do2(self):
        self.observations.sort()
        print('CI begin')
        observations = copy.deepcopy(self.observations)
        print('observations:', observations)
        do_buff = []
        num_share = 3
        episode_num = 0
        s_lst, a_lst, sp_lst, r_lst = self.offdata

        state_moved = torch.autograd.Variable(
            torch.tensor(s_lst).float()).cuda(
            device=self.cuda_device)
        sp_moved = torch.autograd.Variable(
            torch.tensor(sp_lst).float()).cuda(
            device=self.cuda_device)
        r_label_moved = torch.autograd.Variable(torch.from_numpy(np.array(r_lst)).float()).cuda(
            device=self.cuda_device)

        while(1):
            if len(observations)>num_share:
                print('(coarse:) rest observations:', observations)
                do_l = copy.deepcopy(observations)
                spliced_obs = []
                length_each_share = int((len(do_l) - num_share) / num_share)
                share_ind = 0
                for j in range(num_share):
                    share_end = share_ind + length_each_share + 1
                    spliced_obs.append(do_l[share_ind: min(share_end, len(do_l))])
                    share_ind = copy.deepcopy(share_end)
                for gg in range(num_share):
                    count_loss = 0
                    current_obs = spliced_obs[gg]
                    gate_0 = np.zeros([len(s_lst), self.zeros_length, 2])
                    for i in range(len(s_lst)):
                        for j in range(self.zeros_length):
                            if s_lst[i][j] not in current_obs and s_lst[i][j] not in do_buff:
                                gate_0[i][j][1] = 1
                            else:
                                gate_0[i][j][0] = 1

                    for kk in range(1000):
                        h = torch.autograd.Variable(to_tensor(np.zeros((len(s_lst), 800)))).cuda(device=self.cuda_device)
                        c = torch.autograd.Variable(to_tensor(np.zeros((len(s_lst), 800)))).cuda(device=self.cuda_device)
                        for i in range(self.zeros_length):
                            _, h2, c2 = self.forward_pr(state_moved[:, i], h, c)
                            if i == 0:
                                a_pr = self.forward_pr(sp_moved[:, i], h2, c2)[0].unsqueeze(1)
                            else:
                                a_pr = torch.cat((a_pr,self.forward_pr(sp_moved[:, i], h2, c2)[0].unsqueeze(1)), 1)
                            gate = np.zeros([len(s_lst), 2])
                            for j in range(len(s_lst)):
                                gate[j][0] = gate_0[j][i][0]
                                gate[j][1] = gate_0[j][i][1]

                            gate = torch.tensor(gate).float().cuda(device=self.cuda_device).detach()
                            h = torch.mul(gate[:, 0].unsqueeze(1),h) + torch.mul(gate[:, 1].unsqueeze(1),h2)
                            c = torch.mul(gate[:, 0].unsqueeze(1),c) + torch.mul(gate[:, 1].unsqueeze(1),c2)

                        loss = F.smooth_l1_loss(a_pr, r_label_moved)*100
                        if kk != 0:
                            if abs(loss_old - loss.data.cpu().numpy()) <= 0.000005:
                                count_loss += 1
                            else:
                                count_loss = 0
                        if count_loss >= 15:
                            count_loss = 0
                            break
                        loss_old = loss.data.cpu().numpy()
                        self.optimizer_inv.zero_grad()
                        loss.backward()
                        self.optimizer_inv.step()

                        if loss.data.cpu().numpy() < [0.1, 0.01][0 if self.map_id<=1 else 1]:
                            for each_ob in current_obs:
                                if each_ob not in do_buff:
                                    do_buff.append(each_ob)
                            break
                    observations = [ob for ob in observations if ob not in do_buff]
                    print('(coarse:) number of eliminated observations:', len(do_buff))

                    print('(coarse:) number of rest observations:', len(observations))
            else:
                print('(fine:) rest observations:', observations)
                do_l = copy.deepcopy(observations)
                for gg in range(len(do_l)):
                    count_loss = 0
                    gate_0 = np.zeros([len(s_lst), self.zeros_length, 2])
                    for i in range(len(s_lst)):
                        for j in range(self.zeros_length):
                            if s_lst[i][j] != do_l[gg] and s_lst[i][j] not in do_buff:
                                gate_0[i][j][1] = 1
                            else:
                                gate_0[i][j][0] = 1
                    for kk in range(1000):
                        h = torch.autograd.Variable(to_tensor(np.zeros((len(s_lst), 800)))).cuda(
                            device=self.cuda_device)
                        c = torch.autograd.Variable(to_tensor(np.zeros((len(s_lst), 800)))).cuda(
                            device=self.cuda_device)
                        for i in range(self.zeros_length):
                            _, h2, c2 = self.forward_pr(state_moved[:, i], h, c)
                            if i == 0:
                                a_pr = self.forward_pr(sp_moved[:, i], h2, c2)[0].unsqueeze(1)
                            else:
                                a_pr = torch.cat((a_pr, self.forward_pr(sp_moved[:, i], h2, c2)[0].unsqueeze(1)),
                                                 1)
                            gate = np.zeros([len(s_lst), 2])
                            for j in range(len(s_lst)):
                                gate[j][0] = gate_0[j][i][0]
                                gate[j][1] = gate_0[j][i][1]

                            gate = torch.tensor(gate).float().cuda(device=self.cuda_device).detach()
                            h = torch.mul(gate[:, 0].unsqueeze(1), h) + torch.mul(gate[:, 1].unsqueeze(1), h2)
                            c = torch.mul(gate[:, 0].unsqueeze(1), c) + torch.mul(gate[:, 1].unsqueeze(1), c2)

                        loss = F.smooth_l1_loss(a_pr, r_label_moved) * 100
                        if kk != 0:
                            if abs(loss_old - loss.data.cpu().numpy()) <= 0.000005:
                                count_loss += 1
                            else:
                                count_loss = 0
                        if count_loss >= 15:
                            count_loss = 0
                            break
                        loss_old = loss.data.cpu().numpy()
                        self.optimizer_inv.zero_grad()
                        loss.backward()
                        self.optimizer_inv.step()

                        if loss.data.cpu().numpy() < [0.1, 0.01][0 if self.map_id<=1 else 1]:
                            if do_l[gg] not in do_buff:
                                do_buff.append(do_l[gg])
                            break
                    observations = [ob for ob in observations if ob not in do_buff]
                    print('(fine:) number of eliminated observations:', len(do_buff))

                    print('(fine:) number of rest observations:', len(observations))
                print('the end')
                break
        print('gatebuff:', do_buff)
        gate_buff = []
        for sadd in range(len(do_buff)):
            gate_buff.append(do_buff[sadd].index(1))
        self.gatebuff = copy.deepcopy(gate_buff)

    # def set_offdata_cp(self, batch):
    #     s_lst, a_lst, sp_lst, r_lst = batch
    #
    #     if len(self.offdata) == 0:
    #         self.offdata = [[[]], [[]], [[]], [[]]]
    #         for ind in range(self.zeros_length):
    #             self.offdata[0][-1].append(self.add_vec)
    #             self.offdata[1][-1].append(1)
    #             self.offdata[2][-1].append(self.add_vec)
    #             self.offdata[3][-1].append([0,1])
    #
    #     else:
    #         length_off = len(self.offdata[0])
    #         for ind in range(len(s_lst)):
    #             #s_lst_ind = [int(i) for i in s_lst[ind]]
    #             #sp_lst_ind = [int(i) for i in sp_lst[ind]]
    #             if s_lst[ind] != sp_lst[ind]:
    #                 if len(self.offdata[0]) == length_off:
    #                     for i in range(4):
    #                         self.offdata[i].append([])
    #                 self.offdata[0][-1].append(s_lst[ind])
    #                 self.offdata[1][-1].append(a_lst[ind])
    #                 self.offdata[2][-1].append(sp_lst[ind])
    #                 r_label = [0, 0]
    #                 r_label[int(r_lst[ind])] = 1
    #                 self.offdata[3][-1].append(r_label)
    #         if len(self.offdata[0][-1])>=self.zeros_length-2: #or \
    #             #len(self.offdata[0][-1]) <= 5:
    #             for i in range(4):
    #                 self.offdata[i]=self.offdata[i][0:-1]#.pop()
    #         else:
    #             while len(self.offdata[0][-1]) < self.zeros_length:
    #                 self.offdata[0][-1].append(self.add_vec)
    #                 self.offdata[1][-1].append(1)
    #                 self.offdata[2][-1].append(self.add_vec)
    #                 self.offdata[3][-1].append([0,1])
    #         #print(self.offdata[0][-1], self.offdata[0][0:-1])
    #         #print('0:', len(self.offdata[0]))
    #         if len(self.offdata[0]) >= 2:
    #             if self.offdata[0][-1] in self.offdata[0][0:-1]:
    #                 for i in range(4):
    #                     self.offdata[i]=self.offdata[i][0:-1]
    #         # print('1:', len(self.offdata[0]))
    #     for s in batch[0]:
    #         s = [int(i) for i in s]
    #         #print(s, self.observations)
    #         if s not in self.observations:
    #             self.observations.append(s)
    #     self.observations.sort()
