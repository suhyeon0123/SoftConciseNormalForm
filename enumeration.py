from queue import PriorityQueue
from util2 import *

sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()

w = PriorityQueue()

scanned = set()

w.put((REGEX().getCost(), REGEX()))


count = 0
traversed = 1
start = time.time()
prevCost = 0

rpn9 = 0
rpn10 = 0
rpn11 = 0
rpn12 = 0
rpn13 = 0
rpn14 = 0
rpn15 = 0
rpn16 = 0


finished = False


while not w.empty() and not finished:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]

    prevCost = cost
    hasHole = s.hasHole()

    #print("state : ", s, " cost: ",cost)
    if hasHole and s.rpn()<=14:
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


            if k.starnormalform():
                #print(repr(k), "starNormalForm")
                continue

            if k.redundant_concat1():
                #print("concat1")
                continue

            if k.redundant_concat2():
                #print("concat2")
                continue

            if k.KCK():
                # print(repr(k), "is kc_qc")
                continue

            if k.KCQ():
                # print(repr(k), "is kc_qc")
                continue

            if k.QC():
                # print(repr(k), "is kc_qc")
                continue

            '''if ('(00)?0?' in repr(k)) or ('(11)?1?' in repr(k)) or ('0?(00)?' in repr(k)) or ('1?(11)?' in repr(k)) or ('(000?)*' in repr(k)) or ('(111?)*' in repr(k)):
                #print(repr(k), "is concatQ")
                continue'''

            if type(new_elem) == type(Question()) and k.OQ():
                #print(repr(k), "is OQ")
                continue

            if is_orinclusive(k):
                #print(repr(k), "is orinclusive")
                continue

            if k.prefix():
                #print(repr(k), "is prefix")
                continue

            if k.sigmastar():
                #print(repr(k), "is equivalent_KO")
                continue


            w.put((k.getCost(), k))



    if count % 1000 == 0:
        print("Iteration:", count, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
    count = count+1

    if s.rpn()<=16:
        rpn16 +=1
    if s.rpn() <= 15:
        rpn15 += 1
    if s.rpn()<=14:
        rpn14 +=1
    if s.rpn() <= 13:
        rpn13 += 1
    if s.rpn() <= 12:
        rpn12 += 1
    if s.rpn() <= 11:
        rpn11 += 1
    if s.rpn() <= 10:
        rpn10 += 1
    if s.rpn() <= 9:
        rpn9 += 1

print("--end--")
print("count = ")
print(count)
print("rpn9: "+ str(rpn9))
print("rpn10: "+ str(rpn10))
print("rpn11: "+ str(rpn11))
print("rpn12: "+ str(rpn12))
print("rpn14: "+ str(rpn14))
print("rpn13: "+ str(rpn13))
print("rpn14: "+ str(rpn14))
print("rpn15: "+ str(rpn15))
print("rpn16: "+ str(rpn16))

print("time = ")
print(time.time()-start)





