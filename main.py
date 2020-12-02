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

sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()


def is_solution(s, examples):

    if s == '@emptyset':
        return False

    for i in examples.getPos():
        if not str2regexp(s).evalWordP(i):
            return False

    for i in examples.getNeg():
        if str2regexp(s).evalWordP(i):
            return False

    return True

def is_pdead(s, examples):

    s2 = copy.deepcopy(s)
    s2.spreadAll()
    s = repr(s2)

    if s == '@emptyset':
        return True

    for i in examples.getPos():
        if not str2regexp(s).evalWordP(i):
            return True

    return False

def is_ndead(s, examples):

    s2 = copy.deepcopy(s)
    s2.spreadNp()
    s = repr(s2)

    if s == '@emptyset':
        return False

    for i in examples.getNeg():
        if str2regexp(s).evalWordP(i):
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

w.put((1, Character('0')))
w.put((1, Character('1')))
w.put((3, Or(Hole(),Hole())))
w.put((3, Concatenate(Hole(),Hole())))
w.put((2, KleenStar(Hole())))


examples = Examples(3)
answer = examples.getAnswer()

print(examples.getPos(), examples.getNeg())

i = 0
start = time.time()
prevCost = 0

while not w.empty() and i < 10000000:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]
    #print(s, w.qsize(), cost)
    #if cost > prevCost:
    #    scanned.clear()

    prevCost = cost

    hasHole = s.hasHole()

    #and not isPrune(s, examples)
    if hasHole :

        for j, new_elem in enumerate([Character('0'), Character('1'), Character('@epsilon'), Character('@emptyset'), Or(Hole(), Hole()), Concatenate(Hole(), Hole()), KleenStar(Hole())]):

            k = copy.deepcopy(s)

            k.spread(copy.deepcopy(new_elem))

            if k.__repr__ in scanned:
                continue
            else:
                scanned.add(k.__repr__)

            if is_pdead(k, examples):
                #print(repr(k), "is pdead")
                continue

            if is_ndead(k, examples):
                #print(repr(k), "is ndead")
                continue

            if(repr(k) == '0(0+1)*'):
                print(repr(k), cost)


            if j<4:
                w.put((cost, k))
            elif j==4: # Union
                w.put((cost+2, k))
            elif j==5: # Concatenation
                w.put((cost+2, k))
            else: # Kleene Star
                w.put((cost+1, k))


    elif is_solution(repr(s), examples):
        end = time.time()
        print(end-start)
        print("result:", s)
        break
    #else:
    #    print("Not a solution:", s)


    if i % 100 == 0:
        print(i, cost, len(scanned), w.qsize())

    '''if i % 5000 == 4999:
        w = removeOverlap(w)
        print("remove")'''
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)





