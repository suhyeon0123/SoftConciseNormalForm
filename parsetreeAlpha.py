#Regular Expression Implementation ,Written by Adrian Stoll
import copy
import re2 as re
from config import *
from enum import Enum

class Type(Enum):
    HOLE = 0
    CHAR = 1
    K = 2
    Q = 3
    C = 4
    U = 5
    EPS = 6
    REGEX = 10

def is_inclusive(superset, subset):
    # R -> R    sup-nohole sub-nohole
    if repr(superset) == repr(subset) and not superset.hasHole():
        return True
    # R -> R*, R -> R?, R?->R* - nohole
    if (superset.type == Type.K or superset.type == Type.Q) and not superset.hasHole():
        if repr(superset.r) == repr(subset):
            return True
        elif subset.type == Type.Q and repr(superset.r) == repr(subset.r):
            return True


class RE:
    def __lt__(self, other):
        return False

    def rpn(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.rpn()+1
        elif self.type == Type.C or self.type == Type.U:
            return sum(list(i.rpn() for i in self.list)) + len(self.list) -1
        elif self.type == Type.REGEX:
            return self.r.rpn()
        else:
            return 1

    def spread(self, case):
        self.string = None

        if self.type == Type.K or self.type == Type.Q:
            if self.r.type == Type.HOLE:
                self.r = case
                return True
            else:
                return self.r.spread(case)

        elif self.type == Type.C:
            for index, regex in enumerate(self.list):
                if regex.type == Type.HOLE and case.type == Type.C:
                    self.list.append(Hole())
                    return True
                elif regex.type == Type.HOLE:
                    if case.type == Type.CHAR or repr(case) == '0|1':
                        self.checksum = index
                    self.list[index] = case
                    return True
                if self.list[index].spread(case):
                    if not self.list[index].hasHole():
                        self.checksum = index
                    return True
            return False

        elif self.type == Type.U:
            for index, regex in enumerate(self.list):

                if regex.type == Type.HOLE and case.type == Type.U:
                    self.list.append(Hole())
                    self.list.sort(
                        key=lambda regex: '~' if repr(regex) == '#' else ('}' if regex.hasHole() else repr(regex)))
                    return True
                elif regex.type == Type.HOLE:
                    self.list[index] = case
                    self.list.sort(
                        key=lambda regex: '~' if repr(regex) == '#' else ('}' if regex.hasHole() else repr(regex)))
                    return True
                elif self.list[index].spread(case):
                    self.list.sort(
                        key=lambda regex: '~' if repr(regex) == '#' else ('}' if regex.hasHole() else repr(regex)))
                    return True
                else:
                    continue
            return False
        elif self.type == Type.REGEX:
            # 연속된 spread제어
            if case.type != Type.CHAR:
                if case.type == self.lastRE:
                    return False
                if self.lastRE == Type.K and case.type == Type.Q:
                    return False
                if self.lastRE == Type.Q and case.type == Type.K:
                    return False

            if repr(case) == '0|1':
                self.lastRE = Type.CHAR
            else:
                self.lastRE = case.type

            if self.r.type == Type.HOLE:
                self.r = case
                return True
            else:
                return self.r.spread(case)
        else:
            return False

    def spreadAll(self):
        self.string = None

        if self.type == Type.K:
            if self.r.type == Type.HOLE:
                self.r = Or(Character('0'), Character('1'))
            else:
                self.r.spreadAll()
        elif self.type == Type.Q:
            if self.r.type == Type.HOLE:
                self.r = KleenStar(Or(Character('0'), Character('1')))
            else:
                self.r.spreadAll()
        elif self.type == Type.C or self.type == Type.U:
            for index, regex in enumerate(self.list):
                if regex.type == Type.HOLE:
                    self.list[index] = KleenStar(Or(Character('0'), Character('1')))
                else:
                    self.list[index].spreadAll()
        elif self.type == Type.REGEX:
            self.r.spreadAll()

    def spreadNp(self):
        self.string = None
        if self.type == Type.K or self.type == Type.Q:
            if self.r.type == Type.HOLE:
                self.r = Character('@emptyset')
            else:
                self.r.spreadNp()
        elif self.type == Type.C or self.type == Type.U:
            for index, regex in enumerate(self.list):
                if regex.type == Type.HOLE:
                    self.list[index] = Character('@emptyset')
                else:
                    self.list[index].spreadNp()
        elif self.type == Type.REGEX:
            self.r.spreadNp()



    def prior_unroll(self):
        self.string = None

        if self.type == Type.Q:
            self.r.prior_unroll()
        elif self.type == Type.C or self.type == Type.U:
            for index, regex in enumerate(self.list):
                if regex.type == Type.K:
                    s1 = copy.deepcopy(regex.r)
                    s2 = copy.deepcopy(regex.r)
                    s3 = copy.deepcopy(regex)
                    self.list[index] = Concatenate(s1, s2, s3)
                else:
                    self.list[index].prior_unroll()
        elif self.type == Type.REGEX:
            if self.r.type == Type.K:
                s1 = copy.deepcopy(self.r.r)
                s2 = copy.deepcopy(self.r.r)
                s3 = copy.deepcopy(self.r)
                self.r= Concatenate(s1, s2, s3)
            else:
                self.r.prior_unroll()

    def unroll(self):
        if self.type == Type.K:
            s1 = copy.deepcopy(self.r)
            s2 = copy.deepcopy(self.r)
            s3 = copy.deepcopy(self)
            x = Concatenate(s1, s2, s3)
            x.unrolled2 = True
            list = [x]
            list.extend([KleenStar(regex) for regex in self.r.unroll()])
            return list

        elif self.type == Type.Q:
            return [Question(copy.deepcopy(regex)) for regex in self.r.unroll()]
        elif self.type == Type.C or self.type == Type.U:
            result = []
            for index, regex in enumerate(self.list):
                tmp = regex.unroll()
                if len(tmp) > 1:
                    for regex2 in tmp:
                        tmp = copy.deepcopy(self)
                        tmp.list[index] = regex2
                        result.append(tmp)

            if len(result) == 0:
                result.append(copy.deepcopy(self))
            return result
        elif self.type == Type.REGEX:
            return [REGEX(copy.deepcopy(regex)) for regex in self.r.unroll()]
        else:
            return [self]

    def split(self):
        if self.type == Type.K:
            return [copy.deepcopy(self)]
        elif self.type == Type.Q:
            split = []
            split.extend(copy.deepcopy(self.r.split()))
            split.append(copy.deepcopy(Epsilon()))

            return split
        elif self.type == Type.C:
            split = []
            for index, regex in enumerate(self.list):
                sp = regex.split()
                if len(sp) > 1:
                    for splitE in sp:
                        tmp = copy.deepcopy(self)
                        tmp.list[index] = splitE
                        split.append(tmp)

            if len(split) == 0:
                split.append(copy.deepcopy(self))
            return split
        elif self.type == Type.U:
            split = []
            for index, regex in enumerate(self.list):
                if repr(regex) != '#':
                    split.extend(regex.split())
            return split
        elif self.type == Type.REGEX:
            split = []
            rsplit = self.r.split()
            for regex in rsplit:
                split.append(REGEX(regex))
            return split
        else:
            return [self]


    def alpha(self):
        if self.type == Type.K:
            if self.r.type == Type.C and len(self.r.list) == 2:
                if self.r.list[1].type == Type.K and self.r.list[0] == self.r.list[1].r:
                    return True
                if self.r.list[0].type == Type.K and self.r.list[0].r == self.r.list[1]:
                    return True
                if self.r.list[1].type == Type.Q and self.r.list[0] == self.r.list[1].r:
                    return True
                if self.r.list[0].type == Type.Q and self.r.list[0].r == self.r.list[1]:
                    return True

            return self.r.alpha()

        if self.type == Type.Q:
            if self.r.type == Type.C and len(self.r.list) == 2:
                if self.r.list[1].type == Type.K and self.r.list[0] == self.r.list[1].r:
                    return True
                if self.r.list[0].type == Type.K and self.r.list[0].r == self.r.list[1]:
                    return True


            return self.r.alpha()


        elif self.type == Type.C:
            for index, regex in enumerate(self.list):

                if regex.type == Type.K:
                    #x*x?
                    if index+1 < len(self.list) and self.list[index+1].type == Type.Q and repr(regex.r) == repr(self.list[index+1].r):
                        return True
                    #x*x*
                    if index+1 < len(self.list) and self.list[index+1].type == Type.K and repr(regex.r) == repr(self.list[index+1].r):
                        return True
                elif regex.type == Type.Q:
                    #x?x*
                    if index+1 < len(self.list) and self.list[index+1].type == Type.K and repr(regex.r) == repr(self.list[index+1].r):
                        return True

            return any(list(i.alpha() for i in self.list))

        elif self.type == Type.U:

            for index, regex in enumerate(self.list):
                for index2, regex2 in enumerate(self.list):
                    if index < index2 and (is_inclusive(regex, regex2) or is_inclusive(regex2, regex)):
                        return True

            for index, regex in enumerate(self.list):

                if regex.alpha():
                    return True

            return False
        elif self.type == Type.REGEX:
            return self.r.alpha()
        else:
            return False




class Hole(RE):
    def __init__(self):
        self.level = 0
        self.type = Type.HOLE
    def __repr__(self):
        return '#'
    def hasHole(self):
        return True
    def unrolled(self):
        return False
    def getCost(self):
        return HOLE_COST

class REGEX(RE):
    def __init__(self, r = Hole()):
        self.r = r
        self.type = Type.REGEX
        self.lastRE = Type.REGEX
        self.unrolled2 = False
    def __repr__(self):
        return repr(self.r)
    def hasHole(self):
        return self.r.hasHole()
    def unrolled(self):
        return self.r.unrolled()
    def getCost(self):
        return self.r.getCost()

class Epsilon(RE):
    def __init__(self):
        self.level = 0
        self.type = Type.EPS
    def __repr__(self):
        return '@epsilon'
    def hasHole(self):
        return False
    def unrolled(self):
        return False

class Character(RE):
    def __init__(self, c):
        self.c = c
        self.level = 0
        self.type = Type.CHAR
    def __repr__(self):
        return self.c
    def hasHole(self):
        return False
    def unrolled(self):
        return False
    def getCost(self):
        return SYMBOL_COST

class KleenStar(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
        self.type = Type.K
        self.unrolled2 = False

    def __repr__(self):
        if self.string:
            return self.string

        if '{}'.format(self.r) == '@emptyset':
            self.string = '@epsilon'
            return self.string

        if '{}'.format(self.r) == '@epsilon':
            self.string = '@epsilon'
            return self.string


        if self.r.level > self.level:
            self.string = '({})*'.format(self.r)
            return self.string
        else:
            self.string = '{}*'.format(self.r)
            return self.string

    def hasHole(self):
        if not self.hasHole2:
            return False

        if not self.r.hasHole():
            self.hasHole2 = False
        return self.hasHole2

    def unrolled(self):
        if self.unrolled2 or self.r.unrolled():
            return True
        else:
            return False

    def getCost(self):
        return CLOSURE_COST + self.r.getCost()


class Question(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
        self.type = Type.Q
        self.unrolled2 = False

    def __repr__(self):
        if self.string:
            return self.string

        if '{}'.format(self.r) == '@emptyset':
            self.string = '@epsilon'
            return self.string

        if '{}'.format(self.r) == '@epsilon':
            self.string = '@epsilon'
            return self.string

        if self.r.level > self.level:
            self.string = '({})?'.format(self.r)
            return self.string
        else:
            self.string = '{}?'.format(self.r)
            return self.string

    def hasHole(self):
        if not self.hasHole2:
            return False

        if not self.r.hasHole():
            self.hasHole2 = False
        return self.hasHole2

    def unrolled(self):
        if self.unrolled2 or self.r.unrolled():
            return True
        else:
            return False

    def getCost(self):
        return  CLOSURE_COST + self.r.getCost()

class Concatenate(RE):
    def __init__(self, *regexs):
        self.list = list()
        for regex in regexs:
            self.list.append(regex)
        self.level = 2
        self.string = None
        self.hasHole2 = True
        self.checksum = 0
        self.type = Type.C
        self.unrolled2 = False

    def __repr__(self):
        if self.string:
            return self.string

        def formatSide(side):
            if side.level > self.level:
                return '({})'.format(side)
            else:
                return '{}'.format(side)
        for regex in self.list:
            if '@emptyset' in repr(regex):
                self.string = '@emptyset'
                return self.string

        str_list = []
        for regex in self.list:
            if '@epsilon' != repr(regex):
                str_list.append(formatSide(regex))
        return ''.join(str_list)

    def hasHole(self):
        if not self.hasHole2:
            return False

        self.hasHole2 = any(list(i.hasHole() for i in self.list))
        return self.hasHole2
    def unrolled(self):
        if self.unrolled2:
            return True

        self.unrolled2 = any(list(i.unrolled() for i in self.list))
        return self.unrolled2

    def getCost(self):
        return CONCAT_COST + sum(list(i.getCost() for i in self.list))

class Or(RE):
    def __init__(self, a=Hole(), b=Hole()):
        self.list = list()
        self.list.append(a)
        self.list.append(b)
        self.level = 3
        self.string = None
        self.hasHole2 = True
        self.type = Type.U
        self.unrolled2 = False

    def __repr__(self):
        if self.string:
            return self.string

        def formatSide(side):
            if side.level > self.level:
                return '({})'.format(side)
            else:
                return '{}'.format(side)

        str_list = []
        for regex in self.list:
            if repr(regex) != '@emptyset':
                if str_list:
                    str_list.append("|")
                str_list.append(formatSide(regex))
        if not str_list:
            return '@emptyset'
        else:
            return ''.join(str_list)

    def hasHole(self):
        if not self.hasHole2:
            return False

        self.hasHole2 = any(list(i.hasHole() for i in self.list))
        return self.hasHole2

    def unrolled(self):
        if self.unrolled2:
            return True

        self.unrolled2 = any(list(i.unrolled() for i in self.list))
        return self.unrolled2

    def getCost(self):
        return UNION_COST + sum(list(i.getCost() for i in self.list))