from queue import PriorityQueue
import time
from examples import Examples
import configparser
from util import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", type=int,
                    help="Example number")
parser.add_argument("-u", "--unambiguous", help="Set ambiguity",
                    action="store_true")
args = parser.parse_args()


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

            #print(repr(s), repr(new_elem))

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

            if is_redundant(k,examples):
                #print(repr(k), "is redundant")
                continue

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



    if i % 1000 == 0:
        print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)





