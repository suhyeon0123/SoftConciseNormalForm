from queue import PriorityQueue
from util2 import *
import argparse
from examples import*

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", type=int,
                    help="Example number")
args = parser.parse_args()


sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()



w = PriorityQueue()

scanned = set()

w.put((Character('0').getCost(), Character('0')))
w.put((Character('1').getCost(), Character('1')))
w.put((Or().getCost(), Or(isRoot=True)))
w.put((Or(Character('0'),Character('1')).getCost(), Or(Character('0'),Character('1'),isRoot=True)))
w.put((Concatenate(Hole(),Hole()).getCost(), Concatenate(Hole(),Hole(),isRoot=True)))
w.put((KleenStar().getCost(), KleenStar(isRoot=True)))



if args.examples:
    examples = Examples(args.examples)
else:
    examples = Examples(5)
answer = examples.getAnswer()

print(examples.getPos(), examples.getNeg())

i = 0
traversed = 1
start = time.time()
prevCost = 0

finished = False

if is_solution('0', examples, membership):
    print('0')
if is_solution('1', examples, membership):
    print('1')


while not w.empty() and not finished:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]

    prevCost = cost
    hasHole = s.hasHole()

    #print("state : ", s, " cost: ",cost)
    if hasHole:
        for j, new_elem in enumerate([Character('0'), Character('1'), Or(),  Or(Character('0'),Character('1')), Concatenate(Hole(),Hole()), KleenStar(),Question()]):

            #print(repr(s), repr(new_elem))

            k = copy.deepcopy(s)

            if not k.spread(new_elem):
                #print("false")
                continue

            traversed += 1
            if repr(k) in scanned:
                # print("Already scanned?", repr(k))
                # print(list(scanned))
                continue
            else:
                scanned.add(repr(k))


            if type(new_elem)==type(Character('0')) and is_pdead(k, examples):
                #print(repr(k), "is pdead")
                continue

            if (type(new_elem)==type(Character('0')) or type(new_elem)==type(KleenStar()) or type(new_elem)==type(Question())) and is_ndead(k, examples):
                #print(repr(k), "is ndead")
                continue



            if (type(new_elem) == type(KleenStar()) or type(new_elem) == type(Question())) and k.kok():
                #print(repr(k), "is kok")
                continue

            if type(new_elem) == type(Question()) and k.OQ():
                # print(repr(k), "is OQ")
                continue

            if k.kc_qc():
                # print(repr(k), "is kc_qc")
                continue


            checker=False
            if repr(new_elem)=='0|1':
                checker=True
                
            if type(new_elem)==type(Character('0')) or checker:

                if k.prefix():
                    #print(repr(k), "is prefix")
                    continue

                if is_orinclusive(k):
                    #print(repr(k), "is orinclusive")
                    continue

                if k.equivalent_concat():
                    #print(repr(k), "is equivalent_concat")
                    continue

                if is_equivalent_K(k):
                    #print(repr(k), "is equivalent_KO")
                    continue

                if is_new_redundant2(k, examples):
                    #print(repr(k), "is redundant")
                    continue

            print(k)
            if not k.hasHole():
                if is_solution(repr(k), examples, membership):
                    end = time.time()
                    print("Spent computation time:", end-start)
                    print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
                    # print("Result RE:", repr(k), "Verified by FAdo:", is_solution(repr(k), examples, membership2))
                    print("Result RE:", repr(k))
                    finished = True
                    break

            w.put((k.getCost(), k))



    if i % 1000 == 0:
        print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)





