#Regular Expression Implementation ,Written by Adrian Stoll
import copy
import random
import re2 as re
from config import *

def get_rand_re(depth):

    case = random.randrange(0,depth)

    if case > 2:
        cha = True
    else:
        cha = False

    if cha:
        case = random.randrange(0,2)
        if case == 0:
            return Character('0')
        else:
            return Character('1')
    else :
        case = random.randrange(0,5)
        if case <= 0:
            return Or()
        elif case <= 1:
            return Concatenate(Hole(), Hole())
        elif case <= 2:
            return KleenStar()
        elif case <= 3 and depth != 1:
            return Question()
        else:
            return Hole()

class Hole:
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return '#'
    def hasHole(self):
        return True
    def spread(self, case, parentId):
        return False
    def spreadAll(self):
        return KleenStar(Or(Character('0'), Character('1')))
    def spreadRand(self):
        return Character('0') if random.random() <0.5 else Character('1')
    def spreadNp(self):
        return Character('@emptyset')
    def unroll(self):
        return
    def split(self, side):
        return 3
    def make_child(self):
        return
    def overlap(self):
        return False
    def equivalent_K(self):
        return False
    def getn(self):
        return 0
    def orinclusive(self):
        return False
    def kinclusive(self):
        return False
    def singlesymbol(self):
        return False
    def equivalent_concat(self):
        return False
    def equivalent_QCK(self):
        return False
    def kok(self):
        return False
    def or_concat(self):
        return False


class RE:
    def __lt__(self, other):
        return False
    def __init__(self, r=Hole()):
        self.r = r
        self.hasHole2 = True
        self.string = None
        self.first = True
        self.cost = HOLE_COST
        self.holecount=1
        self.lastRE = Hole()

    def __repr__(self):
        if not self.string:
            self.string = repr(self.r)

        return self.string
    def hasHole(self):
        if not self.hasHole2:
            return False
        if not self.r.hasHole():
            self.hasHole2 = False
        return self.hasHole2
    def spread(self, case, parentId):

        if type(case) != type(Character('0')):
            if type(case) == type(self.lastRE):
                return False
            if type(self.lastRE) == type(KleenStar()) and type(case) == type(Question()):
                return False
            if type(self.lastRE) == type(Question()) and type(case) == type(KleenStar()):
                return False
            if type(self.lastRE) == type(KleenStar()) and type(case) == type(Question()):
                return False

        if type(case) == type(Concatenate(Hole(),Hole())):
            self.cost += HOLE_COST + CONCAT_COST
        elif type(case) == type(Or()):
            self.cost += HOLE_COST + UNION_COST
        elif type(case) == type(KleenStar()) or type(case) == type(Question()):
            self.cost += CLOSURE_COST
        else:
            self.cost += - HOLE_COST + SYMBOL_COST

        self.string = None

        if self.first:
            self.r = case
            self.first = False
            self.lastRE = case
            return True
        else:
            x = self.r.spread(case, 10)
            if x:
                self.lastRE = case
            return x

    def spreadAll(self):
        self.string = None
        if type(self.r) == type((Hole())):
            self.r = KleenStar(Or(Character('0'), Character('1')))
        else:
            self.r.spreadAll()
    def spreadRand(self):
        self.string = None
        self.r.spreadRand()
    def spreadNp(self):
        self.string = None
        if type(self.r) == type((Hole())):
            self.r = Character('@emptyset')
        else:
            self.r.spreadNp()
    def unroll(self):
        self.string = None
        self.r.unroll()
    def unroll_entire(self):
        self.string = None
        s1 = copy.deepcopy(self.r.r)
        s2 = copy.deepcopy(self.r.r)
        s3 = copy.deepcopy(self.r)
        self.r = Concatenate(s1,s2,s3)
        for index, regex in enumerate(self.r.list):
            self.r.list[index].unroll()
    def split(self, side):
        self.string = None

        if type(self.r) == type(Or()):
            if repr(self.r.list[side]) != '#':
                self.r = copy.deepcopy(self.r.list[side])
                return 1
            return 2


        return self.r.split(side)
    def make_child(self, count):
        if type(self.r) == type((Hole())):
            self.r = get_rand_re(count)
        else:
            self.r.make_child(count+1)
    def overlap(self):
        return self.r.overlap()
    def equivalent_K(self):
        return self.r.equivalent_K()
    def getn(self):
        return self.r.getn()
    def orinclusive(self):
        return self.r.orinclusive()
    def kinclusive(self):
        return self.r.kinclusive()
    def singlesymbol(self):
        return self.r.singlesymbol()
    def equivalent_concat(self):
        return self.r.equivalent_concat()
    def equivalent_QCK(self):
        return self.r.equivalent_QCK()
    def kok(self):
        return self.r.kok()
    def or_concat(self):
        return self.r.or_concat()

class Epsilon(RE):
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return '@epsilon'
    def hasHole(self):
        return False
    def spread(self, case, parentId):
        return False
    def spreadAll(self):
        return
    def spreadRand(self):
        return
    def spreadNp(self):
        return
    def unroll(self):
        return
    def split(self, side):
        return 3
    def make_child(self, count):
        return
    def overlap(self):
        return False
    def equivalent_K(self):
        return False
    def getn(self):
        return 0
    def orinclusive(self):
        return False
    def kinclusive(self):
        return False
    def singlesymbol(self):
        return False
    def equivalent_concat(self):
        return False
    def equivalent_QCK(self):
        return False
    def kok(self):
        return False
    def or_concat(self):
        return False

class EpsilonBlank(RE):
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return ''
    def hasHole(self):
        return False
    def spread(self, case, parentId):
        return False
    def spreadAll(self):
        return
    def spreadRand(self):
        return
    def spreadNp(self):
        return
    def unroll(self):
        return
    def split(self, side):
        return 3
    def make_child(self, count):
        return
    def overlap(self):
        return False
    def equivalent_K(self):
        return False
    def getn(self):
        return 0
    def orinclusive(self):
        return False
    def kinclusive(self):
        return False
    def singlesymbol(self):
        return False
    def equivalent_concat(self):
        return False
    def equivalent_QCK(self):
        return False
    def kok(self):
        return False


class Character(RE):
    def __init__(self, c):
        self.c = c
        self.level = 0
    def __repr__(self):
        return self.c
    def hasHole(self):
        return False
    def spread(self, case, parentId):
        return False
    def spreadAll(self):
        return self.c
    def spreadRand(self):
        return
    def spreadNp(self):
        return self.c
    def unroll(self):
        return
    def split(self, side):
        return 3
    def make_child(self, count):
        return
    def overlap(self):
        return False
    def equivalent_K(self):
        return False
    def getn(self):
        return 0
    def orinclusive(self):
        return False
    def kinclusive(self):
        return False
    def singlesymbol(self):
        return False
    def equivalent_concat(self):
        return False
    def equivalent_QCK(self):
        return False
    def kok(self):
        return False

class KleenStar(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
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

    def spread(self, case, parentId):
        self.string = None

        if type(self.r)==type((Hole())):
            self.r = case
            return True

        return self.r.spread(case,0)

    def spreadAll(self):
        self.string = None
        if type(self.r) == type((Hole())):
            self.r = Or(Character('0'), Character('1'))
        else:
            self.r.spreadAll()

    def spreadNp(self):
        self.string = None

        if type(self.r) == type(Hole()):
            self.r = Character('@emptyset')
        else:
            self.r.spreadNp()

    def spreadRand(self):
        self.string = None
        if type(self.r) == type(Hole()):
            self.r = Character('0') if random.random() <0.5 else Character('1')
        else:
            self.r.spreadRand()

    def unroll(self):
        self.string = None
        self.r.unroll()

    def split(self, side):
        self.string = None

        if type(self.r) == type(Or()):
            if repr(self.r.list[side]) != '#':
                if type(self.r.list[side]) == type(KleenStar()) or type(self.r.list[side]) == type(Question()):
                    self.r = copy.deepcopy(self.r.list[side].r)
                else:
                    self.r = copy.deepcopy(self.r.list[side])
                return 1
            return 2

        return self.r.split(side)

    def make_child(self, count):
        if type(self.r) == type((Hole())):
            while True:
                x = get_rand_re(count)
                if type(x) != type(KleenStar()) and type(x) != type(Question()):
                    self.r = x
                    break
        else:
            self.r.make_child(count+1)

    def overlap(self):
        return self.r.overlap()

    def equivalent_K(self):
        if not self.hasHole() and repr(self.r) != '1|0':
            return bool(re.fullmatch(repr(self.r), '0')) and bool(re.fullmatch(repr(self.r), '1'))

        if type(self.r)== type(Concatenate('0')):
            for regex in self.r.list:
                if type(regex)==type(KleenStar()) and bool(re.fullmatch(repr(regex.r), '0')) and bool(re.fullmatch(repr(regex.r), '1')):
                    return True

        return self.r.equivalent_K()



    def getn(self):
        return self.r.getn()
    def orinclusive(self):
        return self.r.orinclusive()
    def kinclusive(self):
        if type(self.r) == type(Concatenate()):
            for index, regex in enumerate(self.r.list):

                if type(regex) == type(KleenStar()) and not regex.hasHole():
                    count = 0
                    for regex2 in self.r.list:
                        if repr(regex.r) == repr(regex2) or repr(regex)==repr(regex2):
                            count +=1
                    if count == len(self.r.list):
                        return True

                if type(regex) == type(Question()) and not regex.hasHole():
                    count = 0
                    for regex2 in self.r.list:
                        if repr(regex.r) == repr(regex2) or repr(regex)==repr(regex2):
                            count +=1
                    if count == len(self.r.list):
                        return True

        elif type(self.r) == type(Or()):
            for regex in self.r.list:
                if type(regex) == type(Concatenate()):
                    for regex2 in regex.list:
                        if type(regex2) == type(KleenStar()) and not regex2.hasHole():
                            count = 0
                            for regex3 in regex.list:
                                if repr(regex2.r) == repr(regex3) or repr(regex2)==repr(regex3):
                                    count+=1
                            if count == len(regex.list):
                                #print('type2')
                                return True

                        if type(regex2) == type(Question()) and not regex2.hasHole():
                            count = 0
                            for regex3 in regex.list:
                                if repr(regex2.r) == repr(regex3) or repr(regex2)==repr(regex3):
                                    count+=1
                            if count == len(regex.list):
                                #print('type3')
                                return True

        return self.r.kinclusive()

    def singlesymbol(self):
        if not self.r.hasHole() and type(self.r) != type(Character('0')):
            if '0' not in repr(self.r):
                return True
            if '1' not in repr(self.r):
                return True
        return self.r.singlesymbol()
    def equivalent_concat(self):
        return self.r.equivalent_concat()
    def equivalent_QCK(self):
        return self.r.equivalent_QCK()
    def kok(self):
        if type(self.r)==type(Or()):
            for regex in self.r.list:
                if type(regex) == type(KleenStar()) or type(regex) == type(Question()):
                    return True

        return self.r.kok()



class Question(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
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

    def spread(self, case, parentId):
        self.string = None

        if type(self.r)==type((Hole())):
            if type(case) != type(KleenStar(Hole())) and type(case) != type(Question(Hole())):
                self.r = case
                return True
            else:
                return False

        return self.r.spread(case, 1)

    def spreadAll(self):
        self.string = None

        if type(self.r) == type((Hole())):
            self.r = KleenStar(Or(Character('0'), Character('1')))
        else:
            self.r.spreadAll()

    def spreadRand(self):
        self.string = None
        if type(self.r) == type((Hole())):
            self.r = Character('0') if random.random() <0.5 else Character('1')
        else:
            self.r.spreadRand()

    def spreadNp(self):
        self.string = None

        if type(self.r) == type((Hole())):
            self.r = Character('@emptyset')
        else:
            self.r.spreadNp()

    def unroll(self):
        self.string = None
        self.r.unroll()

    def split(self, side):
        self.string = None

        if type(self.r) == type(Or()):
            if repr(self.r.list[side]) != '#':
                if type(self.r.list[side]) == type(KleenStar()) or type(self.r.list[side]) == type(Question()):
                    self.r = copy.deepcopy(self.r.list[side].r)
                else:
                    self.r = copy.deepcopy(self.r.list[side])
                return 1
            return 2

        return self.r.split(side)

    def make_child(self, count):
        if type(self.r) == type((Hole())) :
            while True:
                x = get_rand_re(count)
                if type(x) != type(Question()) and type(x) != type(KleenStar()):
                    self.r = x
                    break
        else:
            self.r.make_child(count+1)
    def overlap(self):
        return self.r.overlap()
    def equivalent_K(self):
        return self.r.equivalent_K()
    def getn(self):
        return 2
    def orinclusive(self):
        return self.r.orinclusive()
    def kinclusive(self):
        return self.r.kinclusive()
    def singlesymbol(self):
        return self.r.singlesymbol()
    def equivalent_concat(self):
        return self.r.equivalent_concat()
    def equivalent_QCK(self):
        #Q안에 x*, x?, x중 두개의 regex가 concat되어있을때
        if type(self.r) == type(Concatenate()) and len(self.r.list) == 2:
            if (type(self.r.list[0]) == type(KleenStar()) or type(self.r.list[0]) == type(Question()) ) and not self.r.list[0].hasHole():
                if repr(self.r.list[0].r) == repr(self.r.list[1]):
                    return True
            if (type(self.r.list[1]) == type(KleenStar()) or type(self.r.list[1]) == type(Question()) )and not self.r.list[1].hasHole():
                if repr(self.r.list[1].r) == repr(self.r.list[0]):
                    return True

        #QC에 모든 x가 kleene이거나, 모든 x가 Q일때
        if type(self.r) ==type(Concatenate()):
            count = 0
            count2 = 0
            for regex in self.r.list:
                if type(regex)==type(KleenStar()):
                    count +=1
                if type(regex)==type(Question()):
                    count2 +=1
            if count == len(self.r.list) or count2 == len(self.r.list):
                return True

        return self.r.equivalent_QCK()
    def kok(self):
        if type(self.r) == type(Or()):
            for regex in self.r.list:
                if type(regex) == type(KleenStar()) or type(regex) == type(Question()):
                    return True

        return self.r.kok()

class Concatenate(RE):
    def __init__(self, *regexs):
        self.list = list()
        for regex in regexs:
            self.list.append(regex)
        self.level = 2
        self.string = None
        self.hasHole2 = True
        self.checksum = 0

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

    def spread(self, case, parentId):
        self.string = None

        for index, regex in enumerate(self.list):
            if type(regex) == type(Hole()) and type(case) == type(Concatenate(Hole(),Hole())):
                self.list.append(Hole())
                return True
            elif type(regex) == type(Hole()):
                if type(case) == type(Character('0')):
                    self.checksum = index
                self.list[index] = case
                return True
            if self.list[index].spread(case, 2):
                if not self.list[index].hasHole():
                    self.checksum = index
                return True

        return False

    def spreadAll(self):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type(Hole()):
                self.list[index] = KleenStar(Or(Character('0'), Character('1')))
            else:
                self.list[index].spreadAll()

    def spreadRand(self):
        self.string = None
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type((Hole())):
                self.list[index] = Character('0') if random.random() < 0.5 else Character('1')
            else:
                self.list[index].spreadRand()

    def spreadNp(self):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type(Hole()):
                self.list[index] = Character('@emptyset')
            else:
                self.list[index].spreadNp()

    def unroll(self):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type(KleenStar()) and type(regex.r) != type(Hole()):
                a = copy.deepcopy(regex.r)
                b = copy.deepcopy(regex.r)
                c = copy.deepcopy(regex)
                self.list[index] = Concatenate(a,b,c)
                self.list[index].list[0].unroll()
                self.list[index].list[1].unroll()
                self.list[index].list[2].r.unroll()
            else:
                self.list[index].unroll()

    def split(self, side):
        self.string = None

        for index, regex in enumerate(self.list):
            if type(regex) == type(Or()):
                if repr(regex.list[side]) != '#':
                    self.list[index] = copy.deepcopy(self.list[index].list[side])
                    #self.list[index] = regex.list[side]
                    return 1
                return 2

            elif type(regex) == type(Question()):
                if side == 0:
                    if repr(regex.r) != '#':
                        self.list[index] = copy.deepcopy(self.list[index].r)
                        #self.list[index] = regex.r
                    else:
                        return 2
                else:
                    self.list[index] = Epsilon()
                return 1

            if self.list[index].split(side)==1:
                return 1
            if self.list[index].split(side)==2:
                return 2

        return 3

    def make_child(self, count):
        for index, regex in enumerate(self.list):
            if type(regex) == type((Hole())):
                self.list[index] = get_rand_re(count)
            else:
                self.list[index].make_child(count + 1)

    def overlap(self):
        return any(list(i.overlap() for i in self.list))
    def equivalent_K(self):
        for regex in self.list:
            if regex.equivalent_K():
                return True
        return False

    def getn(self):
        for index, regex in enumerate(self.list):
            tmp = regex.getn()
            if tmp != 0:
                return tmp
        return 0
    def orinclusive(self):
        return any(list(i.orinclusive() for i in self.list))
    def kinclusive(self):
        return any(list(i.kinclusive() for i in self.list))
    def singlesymbol(self):
        return any(list(i.singlesymbol() for i in self.list))
    def equivalent_concat(self):
        if self.checksum != 0:
            if (not (type(self.list[self.checksum-1]) == type(Question()) and type(self.list[self.checksum]) == type(Question()) ) and type(self.list[self.checksum-1]) == type(Question()) ) or type(self.list[self.checksum-1]) == type(KleenStar()):
                if type(self.list[self.checksum]) == type(KleenStar()) or type(self.list[self.checksum]) == type(Question()):
                    if repr(self.list[self.checksum-1].r) == repr(self.list[self.checksum].r):
                        self.checksum = 0
                        return True
                else:
                    if repr(self.list[self.checksum - 1].r) == repr(self.list[self.checksum]):
                        self.checksum = 0
                        return True



        return any(list(i.equivalent_concat() for i in self.list))


        '''#sort by [none, Question(), Kleene()]
        for index, regex in enumerate(self.list):
            if index != len(self.list)-1:
                if not self.list[index+1].hasHole():
                    if (type(self.list[index]) != type(KleenStar()) or type(self.list[index]) != type(Question()) )  and (type(self.list[index+1]) == type(KleenStar()) or type(self.list[index+1]) == type(Question()) ) and repr(self.list[index]) == repr(self.list[index + 1].r):
                        return True
                    elif type(self.list[index]) == type(Question())  and type(self.list[index+1]) == type(KleenStar()) and repr(self.list[index].r) == repr(self.list[index + 1].r):
                        return True

        for index, regex in enumerate(self.list):
            if index != len(self.list)-1:
                if type(regex) == type(KleenStar()) and not regex.hasHole() and repr(regex) == repr(self.list[index + 1]):
                    return True
                if type(regex) == type(Question()) and not regex.hasHole() and type(self.list[index + 1]) == type(KleenStar()) and repr(regex.r) == repr(self.list[index + 1].r):
                    return True

            if self.list[index].equivalent_concat():
                return True
        return False'''
    def equivalent_QCK(self):
        return any(list(i.equivalent_QCK() for i in self.list))
    def kok(self):
        return any(list(i.kok() for i in self.list))



class Or(RE):
    def __init__(self, a=Hole(), b=Hole()):
        self.list = list()
        self.list.append(a)
        self.list.append(b)
        self.level = 3
        self.string = None
        self.hasHole2 = True

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



    def spread(self, case, parentId):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type(Hole()) and type(case)==type(Or()):
                self.list.append(Hole())
                self.list.sort(key=lambda regex: '!' if repr(regex) == '#' else ('#' if regex.hasHole() else repr(regex)), reverse=True)
                return True
            elif type(regex)==type((Hole())):
                self.list[index] = case
                self.list.sort(key=lambda regex: '!' if repr(regex) == '#' else ('#' if regex.hasHole() else repr(regex)), reverse=True)
                return True
            if self.list[index].spread(case, 3):
                self.list.sort(key=lambda regex: '!' if repr(regex) == '#' else ('#' if regex.hasHole() else repr(regex)), reverse=True)
                return True
        return False

    def spreadAll(self):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex)==type(Hole()):
                self.list[index] = KleenStar(Or(Character('0'), Character('1')))
            else:
                self.list[index].spreadAll()

    def spreadNp(self):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type(Hole()):
                self.list[index] = Character('@emptyset')
            else:
                self.list[index].spreadNp()
    def spreadRand(self):
        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type((Hole())):
                self.list[index] = Character('0') if random.random() < 0.5 else Character('1')
            else:
                self.list[index].spreadRand()

    def unroll(self):

        self.string = None
        for index, regex in enumerate(self.list):
            if type(regex) == type(KleenStar()) and type(regex.r)!=type(Hole()):
                a = copy.deepcopy(regex.r)
                b = copy.deepcopy(regex.r)
                c = copy.deepcopy(regex)
                self.list[index] = Concatenate(a, b, c)
                self.list[index].list[0].unroll()
                self.list[index].list[1].unroll()
                self.list[index].list[2].r.unroll()
            else:
                self.list[index].unroll()

    def split(self, side):
        return

    def make_child(self, count):
        for index, regex in enumerate(self.list):
            if type(regex) == type((Hole())):
                self.list[index] = get_rand_re(count)
            else:
                self.list[index].make_child(count + 1)

    def overlap(self):
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

    def equivalent_K(self):
        return any(list(i.equivalent_K() for i in self.list))

    def getn(self):
        return len(self.list)

    def orinclusive(self):

        single0 = False
        single1 = False

        for index, regex in enumerate(self.list):

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

            if type(regex) == type(KleenStar()) and not regex.hasHole():
                for regex2 in self.list:

                    if repr(regex.r) == repr(regex2):
                        return True
                    if type(regex2) == type(Question()) and repr(regex.r) == repr(regex2.r):
                        return True

                    # 0101+(01)* 를 구현하려고 함..
                    '''if type(regex2) == type(Concatenate()):
                        pass'''

            if type(regex) == type(Question()) and not regex.hasHole():
                for regex2 in self.list:
                    if repr(regex.r) == repr(regex2):
                        return True

            if type(regex) == type(KleenStar()) and type(regex.r) == type(Or()):
                for inor in regex.r.list:
                    if not inor.hasHole():
                        for x in self.list:
                            if repr(x) == repr(inor):
                                #print('x')
                                return True


            #Or-Concat 결합법칙
            if type(regex) == type(Concatenate()):
                for index2, regex2 in enumerate(self.list):
                    if index < index2 and type(regex2) == type(Concatenate()):
                        if repr(regex.list[0]) == repr(regex2.list[0]):
                            return True
                        if repr(regex.list[len(regex.list)-1]) == repr(regex2.list[len(regex2.list)-1]):
                            return True

            if regex.orinclusive():
                return True

        return False
    def kinclusive(self):
        return any(list(i.kinclusive() for i in self.list))
    def singlesymbol(self):
        return any(list(i.singlesymbol() for i in self.list))
    def equivalent_concat(self):
        return any(list(i.equivalent_concat() for i in self.list))
    def equivalent_QCK(self):
        return any(list(i.equivalent_QCK() for i in self.list))
    def kok(self):
        return any(list(i.kok() for i in self.list))










