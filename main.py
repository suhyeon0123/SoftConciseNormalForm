from queue import PriorityQueue
import time
from examples import Examples
#from parsetree import *
import copy
#from prune import *
import sys
import configparser


#import re
from util import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", type=int,
                    help="Example number")
parser.add_argument("-u", "--unambiguous", help="Set ambiguity",
                    action="store_true")
args = parser.parse_args()


if args.unambiguous:
    from unambiguous_parsetree import *
else:
    from parsetree import *


sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']



w = PriorityQueue()

scanned = set()

w.put((int(config['HOLE_COST']), RE()))

examples = Examples(args.examples)
answer = examples.getAnswer()

print(examples.getPos(), examples.getNeg())

i = 0
traversed = 1
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

        for j, new_elem in enumerate([Character('0'), Character('1'), Or(), Concatenate(), KleenStar(), Question()]):

            # print(repr(s), repr(new_elem))

            k = copy.deepcopy(s)

            if not k.spread(new_elem):
                continue

            traversed += 1
            if repr(k) in scanned:
                # print("Already scanned?", repr(k))
                # print(list(scanned))
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
                    print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
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
        print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)

    '''if i % 5000 == 4999:
        w = removeOverlap(w)
        print("remove")'''
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)





