from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
import re2 as re

from FAdo.cfg import *
from xeger import Xeger
from parsetreeFinal import*
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
    return s.sigmastar()

def is_equivalent2(s):
    return s.equivalent2()


def is_orinclusive(s):
    return s.orinclusive()


def is_pdead(s, examples):
    s_spreadAll = s.repr2()

    for string in examples.getPos():
        if not membership(s_spreadAll, string):
            return True
    return False

def is_ndead(s, examples):
    s_spreadNP = s.repr3()

    if s_spreadNP == '@emptyset':
        return False

    for string in examples.getNeg():
        if membership(s_spreadNP, string):
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



def is_new_redundant3(s, examples):

    tmp = s.unroll()
    if len(tmp) == 1:
        unrolllist = tmp
    else:
        unrolllist = list(filter(lambda x: x.unrolled(), tmp))

    #split
    splitlist = []
    for regex in unrolllist:
        splitlist.extend(regex.split())
    list(map(lambda x: x.spreadAll(), splitlist))

    #중복제거
    splitlist = list(map(repr, splitlist))
    splitset = set(splitlist)
    splitlist = list(splitset)


    #check part
    for state in splitlist:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False

def is_new_redundant3(s, examples):
    #unroll
    unrolllist = s.unroll10()

    print("new:" + str(unrolllist))
    #split
    splitlist = []
    for regex in unrolllist:
        splitlist.extend(regex.split())


    list(map(lambda x: x.spreadAll(), splitlist))

    #중복제거
    splitlist = list(map(repr, splitlist))
    splitset = set(splitlist)
    splitlist = list(splitset)

    #check part
    for state in splitlist:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False


def is_new_redundant2(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)

    '''unrolled_state.prior_unroll()

    # unrolled_state = copy.deepcopy(s)

    # split
    split = unrolled_state.split()'''

    split = unrolled_state.unsp()

    list(map(lambda x: x.spreadAll(), split))

    # 중복제거
    split = list(map(repr, split))
    splitset = set(split)
    split = list(splitset)

    # check part
    for state in split:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False

def is_new_redundant4(s, examples):
    tmp = s.repr_unsp()
    unsp = list(i.replace('#','(0|1)*') for _, i in tmp)


    # check part
    for state in unsp:
        is_red = True
        for string in examples.getPos():
            if membership(state, string):
                is_red = False
                break
        if is_red:
            return True
    return False

def redundantAlpha3(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.prior_unroll()
    tmp = unrolled_state.reprAlpha2()
    unsp = list(i.replace('#','(0|1)*') for _, i in tmp)

    #check part
    for state in unsp:
        is_red = True
        for string in examples.getPos():
            if membership(state, string):
                is_red = False
                break
        if is_red:
            return True
    return False
def redundantAlpha33(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.prior_unroll2()
    tmp = unrolled_state.reprAlpha2()
    unsp = list(i.replace('#','(0|1)*') for _, i in tmp)

    #check part
    for state in unsp:
        is_red = True
        for string in examples.getPos():
            if membership(state, string):
                is_red = False
                break
        if is_red:
            return True
    return False

def redundantAlpha(s, examples):

    # unroll
    tmp1 = time.time()
    unrolled_state = copy.deepcopy(s)
    #print("1:"+str(time.time() - tmp1))


    tmp2 = time.time()
    unrolled_state.prior_unroll()
    #print(str(time.time() - tmp2))



    tmp3 = time.time()
    tmp = unrolled_state.reprAlpha()
    #print(str(time.time() - tmp3))


    tmp6 = time.time()
    unsp = list(i.replace('#','(0|1)*') for _, i in tmp)
    #print(str(time.time() - tmp6))


    tmp4 = time.time()
    #check part
    for state in unsp:
        is_red = True
        for string in examples.getPos():
            if membership(state, string):
                is_red = False
                break
        if is_red:
            return True
    #print(str(time.time() - tmp4))
    #print("all: "+str(time.time() - tmp1))
    return False





def redundantNoQ(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.noq_unroll()
    #print("new " + str(unrolled_state))
    tmp = list(lis[1] for lis in unrolled_state.reprAlpha())
    #print(tmp)
    unsp = []
    for item in tmp:
        item_mod = item.replace('#','(0+1)*').replace('**','*').replace('*?','*').replace('+','|')
        unsp.append(item_mod)


    #check part
    for state in unsp:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False

def redundantAlpha2(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.prior_unroll()
    unrolled_state.spreadAll()
    #print(unrolled_state)


    tmp = unrolled_state.reprAlpha()

    #check part
    for _, state in tmp:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False



def redundantAlphayesq(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.prioryesq_unroll()
    #print(unrolled_state)
    tmp = list(lis[1] for lis in unrolled_state.reprNew())
    #print(tmp)
    unsp = []
    for _, item in tmp:
        item_mod = item.replace('#','(0+1)*').replace('**','*').replace('*?','*').replace('+','|')
        unsp.append(item_mod)


    #check part
    for state in unsp:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False

def redundantNew(s, examples):
    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.new_unroll()
    #print("new " + str(unrolled_state))
    tmp = list(lis[1] for lis in unrolled_state.reprNew())
    #print(tmp)
    unsp = []
    for item in tmp:
        item_mod = item.replace('#','(0+1)*').replace('**','*').replace('*?','*').replace('+','|')
        unsp.append(item_mod)


    #check part
    for state in unsp:
        count = 0
        for string in examples.getPos():
            if membership(state, string):
                count = count + 1
        if count == 0:
            return True
    return False



