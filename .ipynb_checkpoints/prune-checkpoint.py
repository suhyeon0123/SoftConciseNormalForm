from FAdo.reex import *
from FAdo.fa import *
# from FAdo.fio import *
from parseTree import *
import copy

def isPrune(s, examples):
    return isPDead(s,examples) or isNDead(s, examples)

def isPDead(s, examples):
    s = copy.deepcopy(s)
    s.spreadAll(KleenStar(Or(Character('0'),Character('1'))))
    it = iter(examples.getPos())
    for i in it:
        if not str2regexp(repr(s)).evalWordP(i):
            return True
    return False

def isNDead(s, examples):
    s = copy.deepcopy(s)
    #s.spreadAll(Epsilon())
    s.spreadNp()
    if not bool(repr(s).strip()):
        return False
    it = iter(examples.getNeg())
    for i in it:
        if str2regexp(repr(s)).evalWordP(i):
            return True
    return False


def isRedundant(s):
    s = unroll(s)
    it = iter(examples.getPos())
    for i in it:
        if not s.evalWordP(i):
            return True
    return False


def split(s) :
    before = [s]
    after = []
    i = 0
    idx = s.find("+")
    if idx != 0:
        x = s.rfind("(", 0, idx)
        split(s)
        split(s)
    else:
        return s


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
