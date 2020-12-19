from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
import re2 as re
from parsetree import *

def membership(regex, string):
    # print(regex)
    # print(regex, string)
    return bool(re.fullmatch(regex, string))

def membership2(regex, string):
    return str2regexp(regex).evalWordP(string)


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
    if type(s.r) == type(KleenStar()) and not s.hasHole():
        s1 = copy.deepcopy(s.r.r)
        s2 = copy.deepcopy(s.r.r)
        s3 = copy.deepcopy(s.r)
        unrolled_state = copy.deepcopy(Concatenate(Concatenate(s1,s2),s3))
    else:
        s_copy = copy.deepcopy(s)
        s_copy.unroll()
        unrolled_state = copy.deepcopy(s_copy)

    #split
    prev = [unrolled_state]
    next = []

    while prev:

        t = prev.pop()
        if '|' in repr(t):

            if type(t.r) == type(Or()):
                s_left = RE(copy.deepcopy(t.r.a))
                s_right = RE(copy.deepcopy(t.r.b))
            else:
                s_left = copy.deepcopy(t)
                s_left.split(0)
                s_left.__repr__()
                s_right = copy.deepcopy(t)
                s_right.split(1)
                s_right.__repr__()

            prev.append(copy.deepcopy(s_left))
            prev.append(copy.deepcopy(s_right))

        else:
            t.spreadAll()
            next.append(copy.deepcopy(t))


    #check part
    for state in next:

        count = 0

        for string in examples.getPos():
            if membership(repr(state), string):
                break
            count = count+1

        if count == len(examples.getPos()):
            return True

    return False
