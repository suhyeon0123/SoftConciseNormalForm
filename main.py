from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
from queue import PriorityQueue
import time
from examples import Examples
from parseTree import *
import copy
from prune import *
import sys
import configparser

import random
import re2 as re
#import re

sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

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




#can compare when String , but object cant compare
def removeOverlap(w) :
    new = PriorityQueue()
    now = w.get()
    new.put(now)
    while not w.empty():
        tmp = w.get()
        if not now.__repr__()==tmp.__repr__():
            now = tmp
            new.put(now)
    return new

#---------------------------


w = PriorityQueue()

scanned = set()

w.put((int(config['SYMBOL_COST']), Character('0')))
w.put((int(config['SYMBOL_COST']), Character('1')))
w.put((int(config['HOLE_COST']) * 2 + int(config['UNION_COST']), Or(Hole(),Hole())))
w.put((int(config['HOLE_COST']) * 2 + int(config['CONCAT_COST']), Concatenate(Hole(),Hole())))
w.put((int(config['HOLE_COST']) + int(config['CLOSURE_COST']), KleenStar(Hole())))
w.put((int(config['HOLE_COST']) + int(config['CLOSURE_COST']), Question(Hole())))


examples = Examples(29)
answer = examples.getAnswer()

print(examples.getPos(), examples.getNeg())

i = 0
start = time.time()
prevCost = 0

finished = False

while not w.empty() and not finished:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]

    prevCost = cost

    hasHole = s.hasHole()

    if hasHole :

        for j, new_elem in enumerate([Character('0'), Character('1'), Or(Hole(), Hole()), Concatenate(Hole(), Hole()), KleenStar(Hole()), Question(Hole())]):

            k = copy.deepcopy(s)

            if not k.spread(new_elem):
                continue

            if repr(k) in scanned:
                continue
            else:
                scanned.add(repr(k))

            if is_pdead(k, examples):
                #print(repr(k), "is pdead")
                continue

            if is_ndead(k, examples):
                #print(repr(k), "is ndead")
                continue

            #if(repr(k) == '0(0+1)*'):
            #    print(repr(k), cost)

            if not k.hasHole():
                if is_solution(repr(k), examples, membership):
                    end = time.time()
                    print("Spent computation time:", end-start)
                    # print("Result RE:", repr(k), "Verified by FAdo:", is_solution(repr(k), examples, membership2))
                    print("Result RE:", repr(k))
                    finished = True
                    break


            if j<2:
                w.put((cost - int(config['HOLE_COST']) + int(config['SYMBOL_COST']), k))
            elif j==2: # Union
                w.put((cost + int(config['HOLE_COST']) + int(config['UNION_COST']) , k))
            elif j==3: # Concatenation
                w.put((cost + int(config['HOLE_COST']) + int(config['CONCAT_COST']) , k))
            else: # Kleene Star
                w.put((cost + int(config['CLOSURE_COST']) , k))


    #elif is_solution(repr(s), examples):
    #    end = time.time()
    #    print(end-start)
    #    print("result:", s)
    #    break
    #else:
    #    print("Not a solution:", s)


    if i % 1000 == 0:
        print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize())

    '''if i % 5000 == 4999:
        w = removeOverlap(w)
        print("remove")'''
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)





