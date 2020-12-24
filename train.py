#import gym
from queue import PriorityQueue
import math
import random
import numpy as np
from collections import namedtuple

import time
import torch
import torch.optim as optim
from util import *
from DQN import*
from examples import Examples
from parsetree import *
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']


# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#---------------------------


Transition = namedtuple('Transition',
                        ('state', 'pos_example', 'neg_example', 'action', 'next_state', 'reward'))



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

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1

LENGTH_LIMIT = 20
EXAMPLE_LENGHT_LIMIT = 100

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = 6
embed_n = 500

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

scanned = set()



#-----------------------------------

def select_action(regex_tensor, pos_tensor, neg_tensor):
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
            a = policy_net(regex_tensor, pos_tensor, neg_tensor)
            return torch.argmax(a).view(-1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


#-----------------------------------

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return torch.FloatTensor([0])
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
    pos_example_batch = torch.cat(batch.pos_example)
    neg_example_batch = torch.cat(batch.neg_example)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch, pos_example_batch, neg_example_batch).view(-1,6).gather(1, action_batch)



    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states, pos_example_batch,neg_example_batch ).view(-1,6).max(1)[0].detach()
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

    return loss




def make_next_state(state, action, examples):

    copied_state = copy.deepcopy(state)

    success = False

    if action==0:
        spread_success = copied_state.spread(Character('0'))
    elif action == 1:
        spread_success = copied_state.spread(Character('1'))
    elif action == 2:
        spread_success = copied_state.spread(Or())
    elif action == 3:
        spread_success = copied_state.spread(Concatenate())
    elif action == 4:
        spread_success = copied_state.spread(KleenStar())
    elif action == 5:
        spread_success = copied_state.spread(Question())

    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_success:
        done = True
        reward = -10
        return copied_state, reward, done, success

    if repr(copied_state) in scanned:
        done = True
        reward = -10
        return copied_state, reward, done, success
    else:
        scanned.add(repr(copied_state))

    if is_pdead(copied_state, examples):
        #print("pd",state)
        #print(examples.getPos())
        done = True
        reward = -10
        return copied_state, reward, done, success

    if is_ndead(copied_state, examples):
        #print("nd",state)
        done = True
        reward = -10
        return copied_state, reward, done, success

    #if is_redundant(copied_state, examples):
    #    #print("rd ",state )
    #    done = True
    #    reward = 0
    #    return copied_state, reward, done, success

    if not copied_state.hasHole():
        done = True
        if is_solution(repr(copied_state), examples, membership):
            success = True
            end = time.time()
            print("Spent computation time:", end - start)
            print("Found solution: ", copied_state, "Solution length: ", len(repr(copied_state)))
            reward = 100 * (LENGTH_LIMIT + 10 - len(repr(copied_state)))
        else:
            reward = -10
    else:
        done = False
        reward = 10

    return copied_state, reward, done, success


def make_embeded(state,examples):

    pos_examples = examples.getPos()
    neg_examples = examples.getNeg()

    word_index = {'0': 1, '1': 2, '(': 3, ')': 4, '?': 5, '*': 6, '|': 7,
                  'X': 8, '#': 9}
    encoded = []
    for c in repr(state):
        try:
            encoded.append(word_index[c])
        except KeyError:
            encoded.append(100)

    encoded += [0] * (LENGTH_LIMIT + 5 - len(encoded))

    regex_tensor = torch.LongTensor(encoded).view(1, LENGTH_LIMIT + 5)

    encoded = []
    for example in pos_examples:
        if len(example) + len(encoded) +1 > EXAMPLE_LENGHT_LIMIT:
            break
        for c in example:
            try:
                encoded.append(word_index[c])
            except KeyError:
                encoded.append(100)
        encoded.append(10)

    encoded += [0] * (EXAMPLE_LENGHT_LIMIT - len(encoded))
    pos_example_tensor = torch.LongTensor(encoded).view(1, EXAMPLE_LENGHT_LIMIT)

    encoded = []
    for example in neg_examples:
        if len(example) + len(encoded) +1 > EXAMPLE_LENGHT_LIMIT:
            break
        for c in example:
            try:
                encoded.append(word_index[c])
            except KeyError:
                encoded.append(100)
        encoded.append(10)

    encoded += [0] * (EXAMPLE_LENGHT_LIMIT - len(encoded))
    neg_example_tensor = torch.LongTensor(encoded).view(1, EXAMPLE_LENGHT_LIMIT)

    return regex_tensor, pos_example_tensor, neg_example_tensor


w = PriorityQueue()

scanned = set()



finished = False
success = False


num_episodes = 1000

i = 0

start = time.time()

loss = 0
reward_sum = 0

for i_episode in range(num_episodes):

    example_num = random.randint(1, 26)

    examples = Examples(2)
    w.put((int(config['HOLE_COST']), RE()))


    while not w.empty() and not finished:
        if success or i > 2000:
            success = False
            start = time.time()
            w.queue.clear()
            scanned.clear()
            i = 0
            break
            w.put((int(config['HOLE_COST']), RE()))
            print("Restart")

        tmp = w.get()
        state = tmp[1]
        cost = tmp[0]

        prevCost = cost

        if not state.hasHole():
            continue

        for t in range(5):
            chosen_action = select_action(*make_embeded(state,examples))
            next_state, reward, done, success = make_next_state(state,chosen_action,examples)

            reward_sum += reward

            memory.push(*make_embeded(state, examples), chosen_action, make_embeded(next_state, examples)[0], torch.FloatTensor([reward]).to(device))

            loss = optimize_model()

            if done and w.qsize() != 0:
                # print("count =",t)
                break

            if i % 100 == 0 and i != 0:
                print("Episode:", i_episode, "\tExample No.:", example_num, "\tIteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tLoss:", format(loss.item(), '.3f'), "\tAvg Reward:", reward_sum / 100)
                reward_sum = 0

            i = i + 1

            for action in range(6):
                copied_state = copy.deepcopy(state)
                if action == 0:
                    if action == chosen_action:
                        nextCost = cost - int(config['HOLE_COST']) + int(config['SYMBOL_COST'])
                        continue

                    spread_result = copied_state.spread(Character('0'))

                    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_result:
                        continue

                    w.put((cost - int(config['HOLE_COST']) + int(config['SYMBOL_COST']), copied_state))
                elif action == 1:
                    if action == chosen_action:
                        nextCost = cost - int(config['HOLE_COST']) + int(config['SYMBOL_COST'])
                        continue

                    spread_result = copied_state.spread(Character('1'))

                    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_result:
                        continue

                    w.put((cost - int(config['HOLE_COST']) + int(config['SYMBOL_COST']), copied_state))
                elif action == 2:
                    if action == chosen_action:
                        nextCost = cost + int(config['HOLE_COST']) + int(config['UNION_COST'])
                        continue

                    spread_result = copied_state.spread(Or())

                    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_result:
                        continue

                    w.put((cost + int(config['HOLE_COST']) + int(config['UNION_COST']), copied_state))
                elif action == 3:
                    if action == chosen_action:
                        nextCost = cost + int(config['HOLE_COST']) + int(config['CONCAT_COST'])
                        continue

                    spread_result = copied_state.spread(Concatenate())

                    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_result:
                        continue

                    w.put((cost + int(config['HOLE_COST']) + int(config['CONCAT_COST']), copied_state))
                elif action == 4:
                    if action == chosen_action:
                        nextCost = cost + int(config['CLOSURE_COST'])
                        continue

                    spread_result = copied_state.spread(KleenStar())

                    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_result:
                        continue

                    w.put((cost + int(config['CLOSURE_COST']), copied_state))
                elif action == 5:
                    if action == chosen_action:
                        nextCost = cost + int(config['CLOSURE_COST'])
                        continue

                    spread_result = copied_state.spread(Question())

                    if len(repr(copied_state)) > LENGTH_LIMIT or not spread_result:
                        continue

                    w.put((cost + int(config['CLOSURE_COST']), copied_state))

            if cost < 0 or cost > 5000:
                print(cost, nextCost, chosen_action, state, next_state)

            cost = nextCost
            # 다음 상태로 이동
            state = next_state

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())




print('Complete')




