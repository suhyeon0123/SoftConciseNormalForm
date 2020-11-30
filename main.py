from FAdo.reex import *
from FAdo.fa import *
from FAdo.fio import *
from queue import PriorityQueue
import time
from examples import Examples
from parseTree import *
import copy
from prune import *
import sys

#sys.setrecursionlimit(50000)


def is_solution(s, examples):
    it = iter(examples.getPos())
    for i in it:
        if not str2regexp(s).evalWordP(i):
            return False

    it = iter(examples.getNeg())
    for i in it:
        if str2regexp(s).evalWordP(i):
            return False

    return True


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



w.put((1, Character('0')))
w.put((1, Character('1')))
w.put((1, Or(Hole(),Hole())))
w.put((1, Concatenate(Hole(),Hole())))
w.put((1, KleenStar(Hole())))

examples = Examples(2)
answer = examples.getAnswer()

i = 0
start = time.time()

while not w.empty() and i < 100000:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]
    if i == 35120:
        print("dd")
    print(s)
    #and not isPrune(s, examples)
    if s.hasHole() :

        k = copy.deepcopy(s)
        k.spread(Character('0'))
        w.put((cost+1,k))

        k = copy.deepcopy(s)
        k.spread(Character('1'))
        w.put((cost+1, k))

        '''k = copy.deepcopy(s)
        k.spread(Epsilon())
        w.put((cost + 1, k))'''

        k = copy.deepcopy(s)
        k.spread(Or(Hole(),Hole()))
        w.put((cost+3, k))

        k = copy.deepcopy(s)
        k.spread(Concatenate(Hole(),Hole()))
        w.put((cost+3, k))

        k = copy.deepcopy(s)
        k.spread(KleenStar(Hole()))
        w.put((cost+3, k))



    elif not s.hasHole() and is_solution(repr(s), examples):
        end = time.time()
        print(end-start)
        print("result:")
        print(s)
        break

    else:
        print(s)

    '''if i % 5000 == 4999:
        w = removeOverlap(w)
        print("remove")'''
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)





