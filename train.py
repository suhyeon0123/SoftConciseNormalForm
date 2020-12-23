#import gym
import math
import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from util import *
from DQN import*
from examples import Examples
from parsetree import *


# GPU를 사용할 경우
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

#---------------------------


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#----------------------

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = 6
embed_n = 500

policy_net = DQN(embed_n, n_actions).to(device)
target_net = DQN(embed_n, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0




#-----------------------------------

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.


            a = policy_net(state)
            return torch.argmax(a).view(-1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


#-----------------------------------

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    '''print(state_batch.shape)
    print(action_batch.shape)
    print(reward_batch.shape)
    print(non_final_next_states.shape)'''

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).view(-1,6).gather(1, action_batch)



    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).view(-1,6).max(1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()






def make_next_state(state, action, examples):

    if action==0:
        state.spread(Character('0'))
    elif action == 1:
        state.spread(Character('1'))
    elif action == 2:
        state.spread(Or())
    elif action == 3:
        state.spread(Concatenate())
    elif action == 4:
        state.spread(KleenStar())
    '''elif action == 5:
        state.spread(Question())'''


    if is_pdead(state, examples):
        print("pd",state)
        print(examples.getPos())
        done = True
        reward = torch.FloatTensor([-0.1])
        return state, reward, done

    if is_ndead(state, examples):
        print("nd",state)
        done = True
        reward = torch.FloatTensor([-0.1])
        return state, reward, done

    if is_redundant(state, examples):
        print("rd ",state )
        done = True
        reward = torch.FloatTensor([-0.1])
        return state, reward, done

    if not state.hasHole():
        done = True
        if is_solution(repr(state), examples, membership):
            reward = torch.FloatTensor([1])
        else:
            reward = torch.FloatTensor([-0.1])
    else:
        done = False
        reward = torch.FloatTensor([-0.01])

    return state, reward, done


def make_embeded(state,examples):
    pos_examples = examples.getPos()
    neg_examples = examples.getNeg()

    word_index = {'0': 1, '1': 2, '(': 3, ')': 4, '?': 5, '*': 6, '|': 7,
                  'X': 8}
    encoded = []
    for c in repr(state):
        try:
            encoded.append(word_index[c])
        except KeyError:
            encoded.append(100)
    encoded.append(9)
    for example in pos_examples:
        for c in example:
            try:
                encoded.append(word_index[c])
            except KeyError:
                encoded.append(100)
        encoded.append(10)
    encoded.append(9)
    for example in neg_examples:
        for c in example:
            try:
                encoded.append(word_index[c])
            except KeyError:
                encoded.append(100)
        encoded.append(10)
    encoded += [0] * (500 - len(encoded))
    a = torch.FloatTensor(encoded)

    b = a.view([1,1,500])
    return b


num_episodes = 1000
for i_episode in range(num_episodes):

    state = RE()
    examples = Examples(2)

    for t in range(1000):
        action = select_action(make_embeded(state,examples))
        next_state, reward, done = make_next_state(state,action,examples)

        # 메모리에 변이 저장
        memory.push(make_embeded(state,examples), action, make_embeded(next_state,examples), reward)
        # 다음 상태로 이동
        state = next_state

        print(repr(state))
        # 최적화 한단계 수행(목표 네트워크에서)
        optimize_model()
        if done:
            print("count =",t)
            break
    #목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')




