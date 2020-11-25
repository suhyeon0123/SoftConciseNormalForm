from FAdo.reex import *
from FAdo.fa import *
from FAdo.fio import *
from queue import PriorityQueue
import time

class Examples(object):

    def __init__(self):
        self.pos = []
        self.neg = []

    def addPos(self, example):
        self.pos.append(example)

    def addNeg(self, example):
        self.neg.append(example)

    def getPos(self):
        return self.pos

    def getNeg(self):
        return self.neg


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



def isPrune(s):
    return isPDead(s) or isNDead(s)

def isPDead(s):
    x = str2regexp(s.replace("#", "(0+1)*"))
    it = iter(examples.getPos())
    for i in it:
        if not x.evalWordP(i):
            return True
    return False

def isNDead(s):
    x = str2regexp(s.replace("#", "@epsilon*"))
    it = iter(examples.getNeg())
    for i in it:
        if x.evalWordP(i):
            return True
    return False


def isRedundant(s):
    s = unroll(s)
    it = iter(examples.getPos())
    for i in it:
        if not s.evalWordP(i):
            return True
    return False

'''def split(s) :
    before = [s]
    after = []
    i = 0
    idx = s.find("+")
    while True:
        
'''



def unroll(s):
    i = 0
    idx = s.find("*")
    while idx != -1:
        x = s.rfind("(", 0, idx)
        mul = s.rfind("*", 0, idx)
        if mul != -1 and x < mul:
            x = s.rfind("(", 0, x - 1)
            x = s.rfind("(", 0, x - 1)
            x = s.rfind("(", 0, x - 1)

        s = s[:idx] + s[x:idx] + s[x:idx] + s[idx:]
        i = idx + 2 * (idx - x) + 1
        idx = s.find("*", i)
    return s


def removeOverlap(w) :
    new = PriorityQueue()
    now = w.get()
    new.put(now)
    while not w.empty():
        tmp = w.get()
        if not now==tmp:
            now = tmp
            new.put(now)
    return new

#---------------------------

'''s="00(0+1(0)*)*(0)*1"
print(s)
i = 0
idx = s.find("*")
while idx != -1:
    x = s.rfind("(", 0, idx)
    mul = s.rfind("*", 0, idx)
    if mul != -1 and x < mul:
        x = s.rfind("(", 0, x - 1)
        x = s.rfind("(", 0, x - 1)
        x = s.rfind("(", 0, x - 1)

    s = s[:idx] + s[x:idx] + s[x:idx] + s[idx:]
    i = idx + 2*(idx - x) + 1
    idx = s.find("*", i)
print(s)
exit()

x = str2regexp("@epsilon*")
print(x.evalWordP("00"))
'''

examples = Examples()

examples.addPos('0')
examples.addPos('00')
examples.addPos('011')


examples.addNeg('1')
examples.addNeg('10')
examples.addNeg('111')



## queue생성 이후 hole삽입
w = PriorityQueue()
w.put((1, '#'))

i = 0
start = time.time()
while not w.empty() and i < 10000000000:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]

    if s.count("#") != 0 :
        idx = s.find('#')
        w.put((cost+1, s[:idx] + '0' + s[idx+1:]))
        w.put((cost + 1, s[:idx] + '1' + s[idx + 1:]))
        #w.put((cost + 1, s[:idx] + '@epsilon' + s[idx + 1:]))
        w.put((cost + 1, s[:idx] + '(#+#)' + s[idx + 1:]))
        w.put((cost + 1, s[:idx] + '##' + s[idx + 1:]))
        w.put((cost + 1, s[:idx] + '(#)*' + s[idx + 1:]))
        print(s)

    elif s.count("#") == 0 and is_solution(s, examples):
        end = time.time()
        print(end-start)
        print(s)
        break

    else:
        print(s)

    if i % 5000 == 4999:
        w = removeOverlap(w)
        print("remove")
    i = i+1

print("count = ")
print(i)




