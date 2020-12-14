from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
import random
import re2 as re
import copy

def membership(regex, string):
    #print(regex)
    # print(regex, string)
    return bool(re.fullmatch(regex, string))

def membership2(regex, string):
    return str2regexp(regex).evalWordP(string)


def is_solution(regex, examples, membership):

    if regex == '@emptyset':
        return False



    for string in examples.getPos():
        while 'X' in string:
            if random.random() > 0.5:
                string = string.replace(string, '0', 1)
            else:
                string = string.replace(string, '1', 1)

        if not membership(regex, string):
            return False

    for string in examples.getNeg():
        while 'X' in string:
            if random.random() > 0.5:
                string = string.replace(string, '0', 1)
            else:
                string = string.replace(string, '1', 1)

        if membership(regex, string):
            return False

    return True

def is_pdead(s, examples):

    s2 = copy.deepcopy(s)
    s2.spreadAll()
    s = repr(s2)

    if s == '@emptyset':
        return True

    for string in examples.getPos():
        while 'X' in string:
            if random.random() > 0.5:
                string = string.replace(string, '0', 1)
            else:
                string = string.replace(string, '1', 1)

        if not membership(s, string):
            return True

    return False

def is_ndead(s, examples):

    s2 = copy.deepcopy(s)
    s2.spreadNp()
    s = repr(s2)

    if s == '@emptyset':
        return False

    for string in examples.getNeg():

        while 'X' in string:
            if random.random() > 0.5:
                string = string.replace(string, '0', 1)
            else:
                string = string.replace(string, '1', 1)

        if membership(s, string):
            return True

    return False

