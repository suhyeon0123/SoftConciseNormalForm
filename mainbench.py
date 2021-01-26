from queue import PriorityQueue
from util import *
import argparse
from examples import*

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", type=int,
                    help="Example number")
parser.add_argument("-u", "--unambiguous", help="Set ambiguity",
                    action="store_true")
args = parser.parse_args()


sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()

answerl = []
countl = []
timel =[]

pdeadl = []
ndeadl= []
kokl= []
overlapl= []
orinclusivel= []
equivalent_kl= []
k_inclusivel= []
equivalent_concatl= []
equivalent_ql= []
redundantl =[]

for num in range(0,500):


    w = PriorityQueue()

    scanned = set()

    w.put((RE().cost, RE()))

    if os.path.isfile("./rand_benchmarks/no"+str(num)+".txt"):
        examples = Examples(num)
    else:
        continue


    print("no"+str(num))
    pdeadl.append(0)
    ndeadl.append(0)
    kokl.append(0)
    overlapl.append(0)
    orinclusivel.append(0)
    equivalent_kl.append(0)
    k_inclusivel.append(0)
    equivalent_concatl.append(0)
    equivalent_ql.append(0)
    redundantl.append(0)

    answer = examples.getAnswer()

    #print(examples.getPos(), examples.getNeg())

    i = 0
    traversed = 1
    start = time.time()
    prevCost = 0

    finished = False

    target_time = time.time() + 90
    while not w.empty() and not finished and time.time() < target_time:
        tmp = w.get()
        s = tmp[1]
        cost = tmp[0]

        prevCost = cost
        hasHole = s.hasHole()

        #print("state : ", s, " cost: ",cost)
        if hasHole:

            for j, new_elem in enumerate([Character('0'), Character('1'), Or(), Or(Character('0'),Character('1')), Concatenate(Hole(),Hole()), KleenStar(), Question()]):

                #print(repr(s), repr(new_elem))

                checker = False
                if repr(new_elem) == '0|1':
                    checker = True

                k = copy.deepcopy(s)

                if i ==0 and type(new_elem)==type(Question()):
                    continue
                elif not k.spread(new_elem):
                    #print("false")
                    continue

                traversed += 1
                if repr(k) in scanned:
                    # print("Already scanned?", repr(k))
                    # print(list(scanned))
                    continue
                else:
                    scanned.add(repr(k))


                prune = False

                if (type(new_elem) == type(KleenStar()) or type(new_elem) == type(Question())) and k.kok():
                    #print(repr(k), "is kok")
                    prune = True
                    kokl[len(pdeadl)-1] = kokl[len(pdeadl)-1]+1
                    #continue

                if type(new_elem)==type(Character('0')) and is_pdead(k, examples):
                    #print(repr(k), "is pdead")
                    prune = True
                    pdeadl[len(pdeadl) - 1] = pdeadl[len(pdeadl) - 1] + 1
                    #continue

                if (type(new_elem)==type(Character('0')) or type(new_elem)==type(KleenStar()) or type(new_elem)==type(Question())) and is_ndead(k, examples):
                    #print(repr(k), "is ndead")
                    prune = True
                    ndeadl[len(pdeadl) - 1] = ndeadl[len(pdeadl) - 1] + 1
                    #continue


                if type(new_elem)==type(Character('0')) or checker:
                    if is_overlap(k):
                        #print(repr(k), "is overlap")
                        prune = True
                        overlapl[len(pdeadl) - 1] = overlapl[len(pdeadl) - 1] + 1
                        #continue

                    if is_orinclusive(k):
                        #print(repr(k), "is orinclusive")
                        prune = True
                        orinclusivel[len(pdeadl) - 1] = orinclusivel[len(pdeadl) - 1] + 1
                        #continue

                    if is_equivalent_K(k):
                        #print(repr(k), "is equivalent_K")
                        prune = True
                        equivalent_kl[len(pdeadl) - 1] = equivalent_kl[len(pdeadl) - 1] + 1
                        #continue

                    if is_kinclusive(k):
                        #print(repr(k), "is kinclusive")
                        prune = True
                        k_inclusivel[len(pdeadl) - 1] = k_inclusivel[len(pdeadl) - 1] + 1
                        #continue

                    if k.equivalent_concat():
                        #print(repr(k), "is equivalent_concat")
                        prune = True
                        equivalent_concatl[len(pdeadl) - 1] = equivalent_concatl[len(pdeadl) - 1] + 1
                        #continue

                    if k.equivalent_QCK():
                        #print(repr(k), "is equivalent_QCK")
                        prune = True
                        equivalent_ql[len(pdeadl) - 1] = equivalent_ql[len(pdeadl) - 1] + 1
                        #continue

                    if is_new_redundant2(k, examples):
                        #print(repr(k), "is redundant")
                        prune = True
                        redundantl[len(pdeadl) - 1] = redundantl[len(pdeadl) - 1] + 1
                        #continue

                if prune:
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

                w.put((k.cost, k))



        if i % 1000 == 0:
            print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
        i = i+1



    if finished:
        answerl.append(True)
    else:
        answerl.append(False)
    countl.append(i)
    timel.append(time.time()-start)

    #print("--end--")
    #print("count = ")
    #print("answer = " + answer)

f = open("answer.txt",'w')
for i in answerl:
    f.write(str(i) + '\n')
f.close()

f = open("count.txt",'w')
for i in countl:
    f.write(str(i) + '\n')
f.close()

f = open("time.txt",'w')
for i in timel:
    f.write(str(i) + '\n')
f.close()

f = open("pdead.txt",'w')
for i in pdeadl:
    f.write(str(i) + '\n')
f.close()

f = open("ndead.txt",'w')
for i in ndeadl:
    f.write(str(i) + '\n')
f.close()

f = open("kok.txt",'w')
for i in kokl:
    f.write(str(i) + '\n')
f.close()

f = open("overlap.txt",'w')
for i in overlapl:
    f.write(str(i) + '\n')
f.close()

f = open("orinclusive.txt",'w')
for i in orinclusivel:
    f.write(str(i) + '\n')
f.close()

f = open("equivalent_k.txt",'w')
for i in equivalent_kl:
    f.write(str(i) + '\n')
f.close()

f = open("k_inclusive.txt",'w')
for i in k_inclusivel:
    f.write(str(i) + '\n')
f.close()

f = open("equivalent_concat.txt",'w')
for i in equivalent_concatl:
    f.write(str(i) + '\n')
f.close()

f = open("equvalent_q.txt",'w')
for i in equivalent_ql:
    f.write(str(i) + '\n')
f.close()

f = open("redundant.txt",'w')
for i in redundantl:
    f.write(str(i) + '\n')
f.close()

