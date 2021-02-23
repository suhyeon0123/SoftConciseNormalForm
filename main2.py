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

w.put((REGEX().getCost(), REGEX()))

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


while not w.empty() and not finished:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]

    prevCost = cost
    hasHole = s.hasHole()

    #print("state : ", s, " cost: ",cost)
    if hasHole:
        for j, new_elem in enumerate([Character('0'), Character('1'), Or(),  Or(Character('0'),Character('1')), Concatenate(Hole(),Hole()), KleenStar(), Question()]):

            #print(repr(s), repr(new_elem))

            k = copy.deepcopy(s)
            if not k.spread(new_elem):
                #print("false "+ new_elem)
                continue

            traversed += 1
            if repr(k) in scanned:
                # print("Already scanned?", repr(k))
                # print(list(scanned))
                continue
            else:
                scanned.add(repr(k))

            checker = False
            if repr(new_elem) == '0|1':
                checker = True

            # Dead Pruning
            if (type(new_elem)==type(Character('0')) or checker) and is_pdead(k, examples):
                #print(repr(k), "is pdead")
                continue

            if (type(new_elem)==type(Character('0')) or type(new_elem)==type(KleenStar()) or type(new_elem)==type(Question()) or checker) and is_ndead(k, examples):
                #print(repr(k), "is ndead")
                continue


            # Equivalent Pruning
            if k.starnormalform():
                # print(repr(k), "starNormalForm")
                continue

            if k.redundant_concat1():
                # print("concat1")
                continue

            if k.redundant_concat2():
                # print("concat2")
                continue

            if k.KCK():
                # print(repr(k), "is kc_qc")
                continue

            if k.KCQ():
                #print(repr(k), "is kc_qc")
                continue

            if k.QC():
                # print(repr(k), "is kc_qc")
                continue

            if ('(00)?0?' in repr(k)) or ('(11)?1?' in repr(k)) or ('0?(00)?' in repr(k)) or ('1?(11)?' in repr(k)) or (
                    '(000?)*' in repr(k)) or ('(111?)*' in repr(k)):
                # print(repr(k), "is concatQ")
                continue

            if type(new_elem) == type(Question()) and k.OQ():
                # print(repr(k), "is OQ")
                continue

            if is_orinclusive(k):
                # print(repr(k), "is orinclusive")
                continue

            if k.prefix():
                # print(repr(k), "is prefix")
                continue

            if k.sigmastar():
                # print(repr(k), "is equivalent_KO")
                continue

            # Redundant Pruning
            '''if is_new_redundant2(k, examples):
                #print(repr(k), "is redundant")
                continue'''

            if is_new_redundant3(k, examples):
                # print(repr(k), "is redundant")
                continue

            #print(k)
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





