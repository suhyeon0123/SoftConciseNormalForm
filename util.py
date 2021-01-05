from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
import re2 as re

from FAdo.cfg import *
from xeger import Xeger
#from parsetree import*
from parsetree_prune import*
import time
from torch.nn.utils.rnn import pad_sequence
import torch


LENGTH_LIMIT = 30
EXAMPLE_LENGHT_LIMIT = 100

def membership(regex, string):
    # print(regex)
    # print(regex, string)
    # print(regex, string)
    return bool(re.fullmatch(regex, string))

def membership2(regex, string):
    return str2regexp(regex).evalWordP(string)

def tensor_to_regex(regex_tensor):
    word_index = {'pad': 0, '0': 1, '1': 2, '(': 3, ')': 4, '?': 5, '*': 6, '|': 7,
                  'X': 8, '#': 9}
    inverse_word_index = {v: k for k, v in word_index.items()}

    regex = ''

    for i in range(regex_tensor.shape[1]):
        index = inverse_word_index[regex_tensor[0, i].item()]

        if index == 'pad':
            break

        regex += str(index)

    return regex


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
        reward = -100
        return copied_state, reward, done, success

    #if repr(copied_state) in scanned:
    #    done = True
    #    reward = -1
    #    return copied_state, reward, done, success

    # 항상 374line
    if is_pdead(copied_state, examples):
        #print(examples.getPos())
        done = True
        reward = -100
        return copied_state, reward, done, success

    if is_ndead(copied_state, examples):
        done = True
        reward = -100
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
            reward = 100 * (LENGTH_LIMIT + 5 - len(repr(copied_state)))
        else:
            reward = -100
    else:
        done = False
        reward = -10

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

    #encoded += [0] * (LENGTH_LIMIT + 5 - len(encoded))

    regex_tensor = torch.LongTensor(encoded)

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

    #encoded += [0] * (EXAMPLE_LENGHT_LIMIT - len(encoded))
    pos_example_tensor = torch.LongTensor(encoded)

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

    #encoded += [0] * (EXAMPLE_LENGHT_LIMIT - len(encoded))
    neg_example_tensor = torch.LongTensor(encoded)

    return regex_tensor, pos_example_tensor, neg_example_tensor



def is_solution(regex, examples, membership):

    if regex == '@emptyset':
        return False

    for string in examples.getPos():
        if not membership(regex, string):
            return False

    for string in examples.getNeg():
        if membership(regex, string):
            return False

    return True

def is_pdead(s, examples):

    s_copy = copy.deepcopy(s)
    s_copy.spreadAll()
    s = repr(s_copy)

    if s == '@emptyset':
        return True

    for string in examples.getPos():
        if not membership(s, string):
            return True
    return False

def is_ndead(s, examples):

    s_copy = copy.deepcopy(s)
    s_copy.spreadNp()
    s = repr(s_copy)

    if s == '@emptyset':
        return False

    for string in examples.getNeg():
        if membership(s, string):
            return True
    return False


def is_redundant(s, examples):
    #unroll

    if type(s.r) == type(KleenStar()):
        unrolled_state = copy.deepcopy(s)
        unrolled_state.unroll_entire()
    else:
        unrolled_state = copy.deepcopy(s)
        unrolled_state.unroll()

    #unrolled_state = copy.deepcopy(s)



    #split
    prev = [unrolled_state]
    next = []

    count = 0
    while prev:
        count +=1
        if count >=20:
            break
        t = prev.pop()
        if '|' in repr(t):


            if type(t.r) == type(Or()):
                s_left = RE(t.r.a)
                s_right = RE(t.r.b)
            elif type(t.r) == type(Question()):
                s_left = RE(t.r.r)
                s_right = RE(Epsilon())
            else:
                s_left = copy.deepcopy(t)
                s_left.split(0)
                s_right = t
                s_right.split(1)

            #deepcopy problem
            prev.append(s_left)
            prev.append(s_right)

        else:
            t.spreadAll()
            next.append(t)


    #check part
    for state in next:
        count = 0
        for string in examples.getPos():
            if membership(repr(state), string):
                count = count +1
        if count == 0:
            return True
    return False





