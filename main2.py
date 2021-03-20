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

deadtime = 0
pruningtime = 0
spreadtime =0
scantime = 0
queuetime = 0
copytime = 0
solutionputtime = 0

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

    print("state : ", s, " cost: ",cost)
    if hasHole:
        for j, new_elem in enumerate([Character('0'), Character('1'), Or(),  Or(Character('0'),Character('1')), Concatenate(Hole(),Hole()), KleenStar(), Question()]):

            #print(repr(s), repr(new_elem))

            k = copy.deepcopy(s)
            if repr(k) == '(0|1)*1(0|1)##':
                print("ddd")

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

            if not k.hasHole():
                if is_solution(repr(k), examples, membership):
                    end = time.time()
                    print("Spent computation time:", end-start)
                    print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
                    # print("Result RE:", repr(k), "Verified by FAdo:", is_solution(repr(k), examples, membership2))
                    print("Result RE:", repr(k))
                    finished = True
                    break


            checker = False
            if repr(new_elem) == '0|1' or new_elem.type == Type.CHAR:
                checker = True


            # Dead Pruning
            if checker and is_pdead(k, examples):
                #print(repr(k), "is pdead")
                continue

            if (new_elem.type==Type.K or new_elem.type==Type.Q or checker) and is_ndead(k, examples):
                #print(repr(k), "is ndead")
                continue


            '''if k.alpha():
                # print("alpha")
                continue'''

            # Equivalent Pruning
            if (new_elem.type == Type.K or new_elem.type == Type.Q) and k.starnormalform():
                # print(repr(k), "starNormalForm")
                continue

            if checker and k.redundant_concat1():
                # print("concat1")
                continue

            if k.redundant_concat2():
                # print("concat2")
                continue

            if checker and k.KCK():
                # print(repr(k), "is kc_qc")
                continue

            if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and k.KCQ():
                #print(repr(k), "KCQ")
                continue

            if checker and k.QC():
                # print(repr(k), "is kc_qc")
                continue


            if new_elem.type == Type.Q and k.OQ():
                # print(repr(k), "is OQ")
                continue

            if checker and is_orinclusive(k):
                # print(repr(k), "is orinclusive")
                continue

            if checker and k.prefix():
                # print(repr(k), "is prefix")
                continue

            if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and k.sigmastar():
                # print(repr(k), "is equivalent_KO")
                continue


            # because of #?



            #print("k: "+str(k))
            '''if (new_elem.type == Type.Q or checker) and redundantAlpha3(k, examples):
                #print(repr(k), "is redundant")
                continue'''

            # Redundant Pruning
            if (new_elem.type == Type.Q or checker) and redundantAlpha3(k, examples):
                #print(repr(k), "is redundant")
                continue





            #print(k)
            if repr(k) == '(0|1)*1(0|1)##':
                print("ddd")



            w.put((k.getCost(), k))


    if i % 1000 == 0:
        print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
        end = time.time()
        print("Spent computation time:", end - start)
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)
print(str(time.time() - start))
print(pruningtime)
print(deadtime)
print(scantime)
print(spreadtime)
print(solutionputtime)
print(copytime)



