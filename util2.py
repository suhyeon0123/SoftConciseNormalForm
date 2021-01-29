from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
import re2 as re

from FAdo.cfg import *
from xeger import Xeger
#from parsetree import*
from parsetree2 import*
import time
from torch.nn.utils.rnn import pad_sequence
import torch


LENGTH_LIMIT = 30
EXAMPLE_LENGHT_LIMIT = 100

def membership(regex, string):
    # print(regex)
    #print(regex, string)
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
        # print("nd",state)
        done = True
        reward = -100
        return copied_state, reward, done, success

    if is_redundant(copied_state, examples):
        #print("rd ",state )
        done = True
        reward = -100
        return copied_state, reward, done, success

    if not copied_state.hasHole():
        done = True
        if is_solution(repr(copied_state), examples, membership):
            success = True
            reward = 100
        else:
            reward = -100
    else:
        done = False
        reward = -10

    return copied_state, reward, done, success


def make_embeded(state, examples, padding=False):
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

    if padding:
        encoded += [0] * (LENGTH_LIMIT - len(encoded))

    regex_tensor = torch.LongTensor(encoded)

    encoded = []
    for example in pos_examples:
        if padding:
            if len(example) + len(encoded) > EXAMPLE_LENGHT_LIMIT:
                break
        for c in example:
            try:
                encoded.append(word_index[c])
            except KeyError:
                encoded.append(100)
        encoded.append(10)

    if padding:
        encoded += [0] * (EXAMPLE_LENGHT_LIMIT - len(encoded))

    pos_example_tensor = torch.LongTensor(encoded)

    encoded = []
    for example in neg_examples:
        if padding:
            if len(example) + len(encoded) > EXAMPLE_LENGHT_LIMIT:
                break
        for c in example:
            try:
                encoded.append(word_index[c])
            except KeyError:
                encoded.append(100)
        encoded.append(10)

    if padding:
        encoded += [0] * (EXAMPLE_LENGHT_LIMIT - len(encoded))

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

def is_overlap(s):
    return s.overlap()

def is_equivalent_K(s):
    return s.equivalent_K()

def is_equivalent2(s):
    return s.equivalent2()


def is_orinclusive(s):
    return s.orinclusive()


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
    #print(s)
    if s == '@emptyset':
        #print("not dead by blank", ss)
        return False

    for string in examples.getNeg():
        if string == '' and membership(s, string):
            #print("dead by blank", ss)
            return True
        elif membership(s, string):
            #print("not dead by blank", ss)
            return True
    #print("not dead by blank", ss)
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

    exception = False

    count = 0
    while prev:
        count +=1
        if count >=3000:
            print("exception")
            exception = True
            print(s)
            break
        t = prev.pop()
        if ('|' or '?') in repr(t):
            n = t.getn()
            if n == -1:
                if type(t.r) == type(Or()):
                    prev.append(RE())
                else:
                    t.split(-1)
                    prev.append(t)

            else:
                for i in range(n):
                    s_split = copy.deepcopy(t)
                    s_split.split(i)
                    prev.append(s_split)

        else:
            t.spreadAll()
            next.append(t)


    #unrolled_state.spreadAll()
    #next = [unrolled_state]

    if exception:
        #print("list ", next)
        return False

    #check part
    for state in next:
        count = 0
        for string in examples.getPos():
            if membership(repr(state), string):
                count = count + 1
        if count == 0:
            #print(next)
            return True
    return False

def is_new_redundant(s, examples):

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

    exception = False

    count = 0
    while prev:
        count +=1
        if count >=3000:
            #print("exception")
            exception = True
            #print(s)
            break
        t = prev.pop()

        if '|' in repr(t) or '?' in repr(t):
            n = t.getn()
            for i in range(n):
                s_split = copy.deepcopy(t)
                if s_split.split(i)==1:
                    prev.append(s_split)

        else:
            t.spreadAll()
            next.append(t)

    #unrolled_state.spreadAll()
    #next = [unrolled_state]

    if exception:
        #print("list ", next)
        return True

    #check part
    for state in next:
        count = 0
        for string in examples.getPos():
            if membership(repr(state), string):
                count = count + 1
        if count == 0:
            #print(next)
            return True
    return False

def is_new_redundant2(s, examples):

    #unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.unroll2()

    #unrolled_state = copy.deepcopy(s)

    #split
    split = unrolled_state.split2()
    list(map(lambda x: x.spreadAll(), split))
    split = list(map(repr, split))
    splitset = set(split)
    split = list(splitset)


    #check part
    for state in split:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            #print(next)
            return True
    return False

def is_new_redundant2(s, examples):

    #unroll
    if s.type == Type.K:
        s1 = copy.deepcopy(s.r)
        s2 = copy.deepcopy(s.r)
        s3 = copy.deepcopy(s)
        unrolled_state = Concatenate(s1, s2, s3)
    else:
        unrolled_state = copy.deepcopy(s)
        unrolled_state.unroll2()

    #unrolled_state = copy.deepcopy(s)

    #split
    split = unrolled_state.split2()
    list(map(lambda x: x.spreadAll(), split))
    split = list(map(repr, split))
    splitset = set(split)
    split = list(splitset)


    #check part
    for state in split:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            #print(next)
            return True
    return False

'''def star_normal_form(s):
    s_copy = copy.deepcopy(s)
    s_copy.spread(Character('0'))
    s_copy.spread(Character('0'))
    s_copy.spread(Character('0'))
    s_copy.spread(Character('0'))
    origin = repr(s_copy.r)
    norm = RE(Black(s_copy.r))
    for i in range(0, 20):
        norm = norm.removeWhite2()
        norm = norm.removeBlack2()
    #print(origin+ "  " + repr(norm))
    if origin != repr(norm):
        return True
    return False
'''


