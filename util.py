from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
import re2 as re
from parsetree import *

from examples import Examples
from FAdo.cfg import *
from xeger import Xeger

def membership(regex, string):
    # print(regex)
    # print(regex, string)
    # print(regex, string)
    return bool(re.fullmatch(regex, string))

def membership2(regex, string):
    return str2regexp(regex).evalWordP(string)


def gen_str():
    str_list = []

    for i in range(random.randrange(1,7)):
        if random.randrange(1,3) == 1:
            str_list.append('0')
        else:
            str_list.append('1')

    return ''.join(str_list)

def rand_example():
    gen = reStringRGenerator(['0', '1'], random.randrange(3, 15), eps=None)
    regex = gen.generate().replace('+', '|')
    print(regex)

    x = Xeger(limit=10)
    pos_size = 10
    pos_example = list()
    for i in range(1000):
        randStr = x.xeger(regex)
        if len(randStr) <= 7 and randStr not in pos_example:
            pos_example.append(randStr)
            if len(pos_example) == 10:
                break

    neg_example = list()
    for i in range(1000):
        random_str = gen_str()
        if not membership(regex, random_str) and random_str not in neg_example:
            neg_example.append(random_str)
            if len(neg_example) == 10:
                break

    examples = Examples(1)
    examples.setPos(pos_example)
    examples.setNeg(neg_example)

    print(examples.getPos(), examples.getNeg())

    return examples


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
    # if there is #|# - infinite loop..
    '''if '#|#' in repr(s):
        unrolled_state = copy.deepcopy(s)
    elif type(s.r) == type(KleenStar()):
        unrolled_state = copy.deepcopy(s)
        unrolled_state.unroll_entire()
    else:
        unrolled_state = copy.deepcopy(s)
        unrolled_state.unroll()'''


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

    while prev:

        t = prev.pop()

        if '|' in repr(t):

            if type(t.r) == type(Or()):
                s_left = RE(t.r.a)
                s_right = RE(t.r.b)
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

