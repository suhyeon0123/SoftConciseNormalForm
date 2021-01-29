#Regular Expression Implementation ,Written by Adrian Stoll
import copy
import random
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
    #EPSB = 7


class Regex:
    def __init__(self, root):
        self.root = root
        self.hasHole2 = True
        self.string = None
        self.first = True
        self.cost = HOLE_COST
        self.holecount = 1
        self.lastRE = Hole()

    def spread(self, case):
        # 연속된 spread제어
        if case.type != Type.CHAR:
            if case.type == self.lastRE.type:
                return False
            if self.lastRE.type == Type.K and case.type == Type.Q:
                return False
            if self.lastRE.type == Type.Q and case.type == Type.K:
                return False

        #cost
        if case.type == Type.K or case.type == Type.Q:
            self.cost += CLOSURE_COST
        elif case.type == Type.C:
            self.cost += HOLE_COST + CONCAT_COST
        elif case.type == Type.U:
            if repr(case) == '0|1':
                self.cost += - HOLE_COST + SYMBOL_COST + SYMBOL_COST +UNION_COST
            else:
                self.cost += HOLE_COST + UNION_COST
        else:
            self.cost += - HOLE_COST + SYMBOL_COST

        self.string = None

        if self.first:
            self.r = case
            self.first = False
            self.lastRE = case.type
            return True
        else:
            x = self.r.spread(case)
            if x:
                self.lastRE = case.type
            return x

    def unroll2(self):
        self.string = None
        if type(self.r)==type(KleenStar()):
            s1 = copy.deepcopy(self.r.r)
            s2 = copy.deepcopy(self.r.r)
            s3 = copy.deepcopy(self.r)
            self.r = Concatenate(s1, s2, s3)
        elif type(self.r) == type(Question()) or type(self.r) == type(Concatenate()) or type(self.r) == type(Or()):
            self.r.unroll2()
    def split2(self):
        x = self.r.split2()
        result = []
        for regex in x:
            result.append(RE(regex))
        return result


class RE:
    def __lt__(self, other):
        return False

    def spread(self, case):
        # 연속된 spread제어
        if case.type != Type.CHAR and self.isRoot:
            if case.type == self.lastRE:
                return False
            if self.lastRE == Type.K and case.type == Type.Q:
                return False
            if self.lastRE == Type.Q and case.type == Type.K:
                return False

        self.lastRE = case.type
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
                    if case.type == Type.CHAR:
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

        else:
            return False

    def spreadAll(self):
        self.string = None

        if self.type == Type.K:
            if self.r.type == Type.HOLE:
                self.r = Or(Character('0'), Character('1'))
            else:
                self.r.spreadAll()
        if self.type == Type.Q:
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

    def unroll2(self):
        self.string = None

        if self.type == Type.Q:
            self.r.unroll2()
        if self.type == Type.C or self.type == Type.U:
            for index, regex in enumerate(self.list):
                if regex.type == Type.K:
                    s1 = copy.deepcopy(regex.r)
                    s2 = copy.deepcopy(regex.r)
                    s3 = copy.deepcopy(regex)
                    self.list[index] = Concatenate(s1, s2, s3)
                else:
                    self.list[index].unroll2()

    def split2(self):
        if self.type == Type.K:
            return [copy.deepcopy(self)]
        elif self.type == Type.Q:
            split = []
            split.extend(copy.deepcopy(self.r.split2()))
            split.append(copy.deepcopy(Epsilon()))

            return split
        elif self.type == Type.C:
            split = []
            for index, regex in enumerate(self.list):
                sp = regex.split2()
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
                    split.extend(regex.split2())
            return split
        else:
            return [self]

    def overlap(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.overlap()
        elif self.type == Type.C:
            return any(list(i.overlap() for i in self.list))

        elif self.type == Type.U:
            noholelist = []
            for regex in self.list:
                if not regex.hasHole():
                    noholelist.append(repr(regex))
            noholeset = set(noholelist)
            if len(noholelist) != len(noholeset):
                return True

            for regex in self.list:
                if regex.overlap():
                    return True
            return False

        else:
            return False

    def equivalent_K(self):
        if self.type == Type.K:
            if not self.hasHole() and repr(self.r) != '0|1':
                return bool(re.fullmatch(repr(self.r), '0')) and bool(re.fullmatch(repr(self.r), '1'))

            if type(self.r) == type(Concatenate('0')):
                for regex in self.r.list:
                    if type(regex) == type(KleenStar()) and bool(re.fullmatch(repr(regex.r), '0')) and bool(
                            re.fullmatch(repr(regex.r), '1')):
                        return True

            return self.r.equivalent_K()
        elif self.type == Type.Q:
            return self.r.equivalent_K()
        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.equivalent_K() for i in self.list))
        else:
            return False

    def prefix(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.prefix()
        elif self.type == Type.C:
            return any(list(i.prefix() for i in self.list))

        elif self.type == Type.U:
            for index1, regex1 in enumerate(self.list):
                if regex1.type == Type.CHAR:
                    for index2, regex2 in enumerate(self.list):
                        if index1 < index2 and regex2.type == Type.CHAR:
                            if repr(regex1) == repr(regex2):
                                return True
                        elif index1 < index2 and regex2.type == Type.C:
                            if repr(regex1) == repr(regex2.list[0]):
                                return True
                            if repr(regex1) == repr(regex2.list[len(regex2.list) - 1]):
                                return True

                if regex1.type == Type.C:
                    for index2, regex2 in enumerate(self.list):
                        if index1 < index2 and regex2.type == Type.C:
                            if repr(regex1.list[0]) == repr(regex2.list[0]):
                                return True
                            if repr(regex1.list[len(regex1.list) - 1]) == repr(regex2.list[len(regex2.list) - 1]):
                                return True

            return any(list(i.prefix() for i in self.list))
        else:
            return False

    def orinclusive(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.orinclusive()
        elif self.type == Type.C:
            return any(list(i.orinclusive() for i in self.list))

        elif self.type == Type.U:
            single0 = False
            single1 = False

            for index, regex in enumerate(self.list):
                #or singlesymbol
                if not regex.hasHole():
                    if '0' not in repr(regex):
                        if single1:
                            return True
                        else:
                            single1 = True
                    elif '1' not in repr(regex):
                        if single0:
                            return True
                        else:
                            single0 = True

                #x* with x?,x
                if type(regex) == type(KleenStar()) and not regex.hasHole():
                    for regex2 in self.list:
                        if repr(regex.r) == repr(regex2):
                            return True
                        if regex2.type == Type.Q and repr(regex.r) == repr(regex2.r):
                            return True

                # x?, x
                if regex.type == Type.Q and not regex.hasHole():
                    for regex2 in self.list:
                        if repr(regex.r) == repr(regex2):
                            return True

                # 0101+(01)* 를 구현하려고 함..
                '''if type(regex2) == type(Concatenate()):
                    pass'''

                #or inclusive recursive 단편적으로 구현
                '''if type(regex) == type(KleenStar()) and type(regex.r) == type(Or()):
                    for inor in regex.r.list:
                        if not inor.hasHole():
                            for x in self.list:
                                if repr(x) == repr(inor):
                                    # print('x')
                                    return True'''


                if regex.orinclusive():
                    return True

            return False
        else:
            return False


    def equivalent_concat(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.equivalent_concat()

        elif self.type == Type.C:
            if self.checksum != 0:
                if self.list[self.checksum - 1].type == Type.Q and not (
                        self.list[self.checksum].type == Type.K or self.list[self.checksum].type == Type.Q):
                    if repr(self.list[self.checksum - 1].r) == repr(self.list[self.checksum]):
                        return True
                elif self.list[self.checksum - 1].type == Type.Q and self.list[self.checksum].type == Type.K:
                    if repr(self.list[self.checksum - 1].r) == repr(self.list[self.checksum].r):
                        return True
                elif self.list[self.checksum - 1].type == Type.K and not (
                        self.list[self.checksum].type == Type.K or self.list[self.checksum].type == Type.Q):
                    if repr(self.list[self.checksum - 1].r) == repr(self.list[self.checksum]):
                        return True
                elif self.list[self.checksum - 1].type == Type.K and self.list[self.checksum].type == Type.Q:
                    if repr(self.list[self.checksum - 1].r) == repr(self.list[self.checksum].r):
                        return True
                elif self.list[self.checksum - 1].type == Type.K and self.list[self.checksum].type == Type.K:
                    if repr(self.list[self.checksum - 1].r) == repr(self.list[self.checksum].r):
                        return True
            return any(list(i.equivalent_concat() for i in self.list))

        elif self.type == Type.U:
            return any(list(i.equivalent_concat() for i in self.list))

        else:
            return False


    def kok(self):
        if self.type == Type.K:
            if type(self.r) == type(Or()):
                for regex in self.r.list:
                    if type(regex) == type(KleenStar()) or type(regex) == type(Question()):
                        return True
            return self.r.kok()
        elif self.type == Type.Q:
            if type(self.r) == type(Or()):
                for regex in self.r.list:
                    if type(regex) == type(KleenStar()) or type(regex) == type(Question()):
                        return True
            return self.r.kok()

        elif self.type == Type.C:
            return any(list(i.kok() for i in self.list))

        elif self.type == Type.U:
            return any(list(i.kok() for i in self.list))
        else:
            return False

    def OQ(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.OQ()
        elif self.type == Type.C:
            return any(list(i.OQ() for i in self.list))
        elif self.type == Type.U:
            for regex in self.list:
                if type(regex) == type(Question()):
                    return True
            return any(list(i.OQ() for i in self.list))
        else:
            return False

    def kc_qc(self):
        if self.type == Type.K:
            if type(self.r) == type(Concatenate()):
                for index, regex in enumerate(self.r.list):

                    #kc안에 x*와 x들이 있을때
                    if type(regex) == type(KleenStar()) and not regex.hasHole():
                        count = 0
                        for regex2 in self.r.list:
                            if repr(regex.r) == repr(regex2) or repr(regex) == repr(regex2):
                                count += 1
                        if count == len(self.r.list):
                            return True
                    #kc안에 xx?
                    '''if type(regex) == type(Question()) and not regex.hasHole():
                        count = 0
                        for regex2 in self.r.list:
                            if repr(regex.r) == repr(regex2) or repr(regex) == repr(regex2):
                                count += 1
                        if count == len(self.r.list):
                            return True'''

            '''elif type(self.r) == type(Or()):
                for regex in self.r.list:
                    if type(regex) == type(Concatenate()):
                        for regex2 in regex.list:
                            if type(regex2) == type(KleenStar()) and not regex2.hasHole():
                                count = 0
                                for regex3 in regex.list:
                                    if repr(regex2.r) == repr(regex3) or repr(regex2) == repr(regex3):
                                        count += 1
                                if count == len(regex.list):
                                    # print('type2')
                                    return True

                            if type(regex2) == type(Question()) and not regex2.hasHole():
                                count = 0
                                for regex3 in regex.list:
                                    if repr(regex2.r) == repr(regex3) or repr(regex2) == repr(regex3):
                                        count += 1
                                if count == len(regex.list):
                                    # print('type3')
                                    return True'''

            return self.r.kc_qc()

        elif self.type == Type.Q:
            # Q안에 x*, x?, x중 두개의 regex가 concat되어있을때
            if type(self.r) == type(Concatenate()) and len(self.r.list) == 2:
                if (type(self.r.list[0]) == type(KleenStar()) or type(self.r.list[0]) == type(Question())) and not \
                        self.r.list[0].hasHole():
                    if repr(self.r.list[0].r) == repr(self.r.list[1]):
                        return True
                if (type(self.r.list[1]) == type(KleenStar()) or type(self.r.list[1]) == type(Question())) and not \
                        self.r.list[1].hasHole():
                    if repr(self.r.list[1].r) == repr(self.r.list[0]):
                        return True

            # QC에 모든 x가 kleene이거나, 모든 x가 Q일때
            if type(self.r) == type(Concatenate()):
                count = 0
                for regex in self.r.list:
                    if type(regex) == type(KleenStar()) or type(regex) == type(Question()):
                        count += 1
                if count == len(self.r.list):
                    return True

            return self.r.kc_qc()

        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.kc_qc() for i in self.list))
        else:
            return False


class Hole(RE):
    def __init__(self, isRoot=False):
        self.level = 0
        self.type = Type.HOLE
        self.isRoot = isRoot
    def __repr__(self):
        return '#'
    def hasHole(self):
        return True
    def getCost(self):
        return HOLE_COST


class Epsilon(RE):
    def __init__(self):
        self.level = 0
        self.type = Type.EPS
    def __repr__(self):
        return '@epsilon'
    def hasHole(self):
        return False

class Character(RE):
    def __init__(self, c, isRoot=False):
        self.c = c
        self.level = 0
        self.type = Type.CHAR
        self.isRoot = isRoot
    def __repr__(self):
        return self.c
    def hasHole(self):
        return False
    def getCost(self):
        return SYMBOL_COST

class KleenStar(RE):
    def __init__(self, r=Hole(), isRoot=False ):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
        self.type = Type.K
        self.lastRE = Type.K
        self.isRoot = isRoot

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
    def getCost(self):
        return CLOSURE_COST + self.r.getCost()

class Question(RE):
    def __init__(self, r=Hole(), isRoot=False):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
        self.type = Type.Q
        self.isRoot = isRoot
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
    def getCost(self):
        return  CLOSURE_COST + self.r.getCost()

class Concatenate(RE):
    def __init__(self, *regexs, isRoot = False):
        self.list = list()
        for regex in regexs:
            self.list.append(regex)
        self.level = 2
        self.string = None
        self.hasHole2 = True
        self.checksum = 0
        self.type = Type.C
        self.lastRE = Type.C
        self.isRoot = isRoot

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
    def getCost(self):
        return CONCAT_COST + sum(list(i.getCost() for i in self.list))

class Or(RE):
    def __init__(self, a=Hole(), b=Hole(), isRoot=False):
        self.list = list()
        self.list.append(a)
        self.list.append(b)
        self.level = 3
        self.string = None
        self.hasHole2 = True
        self.type = Type.U
        self.lastRE = Type.U
        self.isRoot = isRoot

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
    def getCost(self):
        return UNION_COST + sum(list(i.getCost() for i in self.list))