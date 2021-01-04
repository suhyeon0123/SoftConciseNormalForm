#Regular Expression Implementation ,Written by Adrian Stoll
import copy
import configparser
import random


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']


def get_rand_re(depth):

    case = random.randrange(0,2+depth)

    if case > 2:
        cha = True
    else:
        cha = False

    if cha :
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
            return Concatenate()
        elif case <= 2:
            return KleenStar()
        elif case <= 3:
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
    def spread(self, case):
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
        return False
    def make_child(self):
        return



class RE:
    def __lt__(self, other):
        return False
    def __init__(self, r=Hole()):
        self.r = r
        self.hasHole2 = True
        self.string = None
        self.first = True
        self.cost = int(config['HOLE_COST'])

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
    def spread(self, case):

        if type(case) == type(Concatenate()):
            self.cost += int(config['HOLE_COST']) + int(config['CONCAT_COST'])
        elif type(case) == type(Or()):
            self.cost += int(config['HOLE_COST']) + int(config['UNION_COST'])
        elif type(case) == type(KleenStar()) or type(case) == type(Question()):
            self.cost += int(config['HOLE_COST']) + int(config['CLOSURE_COST'])
        else:
            self.cost += -int(config['HOLE_COST']) + int(config['SYMBOL_COST'])

        self.string = None

        if self.first:
            self.r = case
            self.first = False
            return True
        else:
            return self.r.spread(case)
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
        self.string = None
    def unroll_entire(self):
        s1 = copy.deepcopy(self.r.r)
        s2 = copy.deepcopy(self.r)
        r = Concatenate(Concatenate(s1,s1),s2)
    def split(self, side):
        self.r.split(side)
        self.string = None
    def make_child(self, count):
        if type(self.r) == type((Hole())):
            self.r = get_rand_re(count)
        else:
            self.r.make_child(count)


class Epsilon(RE):
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return '@epsilon'
    def hasHole(self):
        return False
    def spread(self, case):
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
        return False
    def make_child(self, count):
        return



class EpsilonBlank(RE):
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return ''
    def hasHole(self):
        return False
    def spread(self, case):
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
        return False
    def make_child(self, count):
        return


class Character(RE):
    def __init__(self, c):
        self.c = c
        self.level = 0
    def __repr__(self):
        return self.c
    def hasHole(self):
        return False
    def spread(self, case):
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
        return False
    def make_child(self, count):
        return


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

    def spread(self, case):
        self.string = None

        if type(self.r)==type((Hole())):
            if type(case) != type(KleenStar(Hole())) and type(case) != type(Question(Hole())):
                self.r = case
                return True
            else:
                return False

        return self.r.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.r) == type((Hole())):
            self.r = Or(Character('0'), Character('1'))
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
        print("ttt")
        if type(self.r) == type(Or()):
            if side == 0:
                if type(self.r.a) == type(KleenStar()):
                    self.r = copy.deepcopy(self.r.a.r)
                else:
                    self.r = copy.deepcopy(self.r.a)
                return True
            else:
                if type(self.r.b) == type(KleenStar()):
                    self.r = copy.deepcopy(self.r.b.r)
                else:
                    self.r = copy.deepcopy(self.r.b)
                return True
        elif type(self.r) == type(Question()):
            if side == 0:
                self.r = copy.deepcopy(self.r.r.r)
                return True
            else:
                self.r = Epsilon()
                return True

        return self.r.split(side)
    def make_child(self, count):
        if type(self.r) == type((Hole())):
            while True:
                x = get_rand_re(count)
                if type(x) != type(KleenStar()) and type(x) != type(Question()):
                    self.r = x
                    break
        else:
            self.r.make_child(count)


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

    def spread(self, case):
        self.string = None

        if type(self.r)==type((Hole())):
            if type(case) != type(KleenStar(Hole())) and type(case) != type(Question(Hole())):
                self.r = case
                return True
            else:
                return False

        return self.r.spread(case)

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
        print("Ddd")
        if type(self.r) == type(Or()):
            self.r = copy.deepcopy(self.r.a) if side == 0 else copy.deepcopy(self.r.b)
            return True
        elif type(self.r) == type(Question()):
            if side == 0:
                self.r = copy.deepcopy(self.r.r.r)
                return True
            else:
                self.r = Epsilon()
                return True
        return self.r.split(side)

    def make_child(self, count):
        if type(self.r) == type((Hole())) :
            while True:
                x = get_rand_re(count)
                if type(x) != type(Question()) and type(x) != type(KleenStar()):
                    self.r = x
                    break
        else:
            self.r.make_child(count)


class Concatenate(RE):
    def __init__(self, a=Hole(), b=Hole()):
        self.a, self.b = a, b
        self.level = 2
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

        if '@emptyset' in repr(self.a) or '@emptyset' in repr(self.b):
            self.string = '@emptyset'
            return self.string

        if '@epsilon' == repr(self.a):
            self.string = formatSide(self.b)
            return self.string
        elif '@epsilon' == repr(self.b):
            self.string = formatSide(self.a)
            return self.string

        self.string = formatSide(self.a) + formatSide(self.b)
        return self.string

    def hasHole(self):

        if not self.hasHole2:
            return False

        self.hasHole2 = self.a.hasHole() or self.b.hasHole()

        return self.hasHole2

    def spread(self, case):
        self.string = None

        if type(self.a)==type((Hole())):
            self.a = case
            return True
        elif type(self.b)==type((Hole())):
            self.b = case
            return True
        if self.a.spread(case):
            return True
        else:
            return self.b.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.a)==type((Hole())):
            self.a = KleenStar(Or(Character('0'), Character('1')))
        else:
            self.a.spreadAll()
        if type(self.b)==type((Hole())):
            self.b = KleenStar(Or(Character('0'), Character('1')))

        else:
            self.b.spreadAll()

    def spreadRand(self):
        self.string = None
        if type(self.a) == type((Hole())):
            self.a = Character('0') if random.random() <0.5 else Character('1')
        else:
            self.a.spreadRand()
        if type(self.b) == type((Hole())):
            self.b = Character('0') if random.random() <0.5 else Character('1')
        else:
            self.b.spreadRand()

    def spreadNp(self):
        self.string = None
        if type(self.a)==type((Hole())):
            self.a = Character('@emptyset')
        else:
            self.a.spreadNp()

        if type(self.b)==type((Hole())):
            self.b = Character('@emptyset')
        else:
            self.b.spreadNp()

    def unroll(self):
        self.string = None
        if type(self.a) == type(KleenStar()) and not self.a.hasHole():
            s = copy.deepcopy(self.a.r)
            self.a = Concatenate(Concatenate(s, s), KleenStar(s))

            self.a.a.unroll()
            self.a.b.unroll()

        else:
            self.a.unroll()

        if type(self.b) == type(KleenStar()) and not self.b.hasHole():
            s = copy.deepcopy(self.b.r)
            self.b = Concatenate(Concatenate(s, s), KleenStar(s))

            self.b.a.unroll()
            self.b.b.unroll()

        else:
            self.b.unroll()


    def split(self, side):
        self.string = None
        print("ttt")
        if type(self.a) == type(Or()):
            self.a = copy.deepcopy(self.a.a) if side == 0 else copy.deepcopy(self.a.b)
            return True
        elif type(self.a) == type(Question()):
            self.a = copy.deepcopy(self.a.r) if side == 0 else Epsilon()
            return True
        elif type(self.b) == type(Or()):
            self.b = copy.deepcopy(self.b.a) if side == 0 else copy.deepcopy(self.b.b)
            return True
        elif type(self.b) == type(Question()):
            self.b = copy.deepcopy(self.b.r) if side == 0 else Epsilon()
            return True

        if self.a.split(side):
            return True
        else:
            return self.b.split(side)

    def make_child(self, count):
        if type(self.a) == type((Hole())):
            self.a = get_rand_re(count)
        else:
            self.a.make_child(count)

        if type(self.b) == type((Hole())):
            self.b = get_rand_re(count)
        else:
            self.b.make_child(count)


class Or(RE):
    def __init__(self, a=Hole(), b=Hole()):
        self.a, self.b = a, b
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

        if repr(self.a) == '@emptyset':
            self.string = formatSide(self.b)
            return self.string
        elif repr(self.b) == '@emptyset':
            self.string = formatSide(self.a)
            return self.string

        self.string = formatSide(self.a) + '|' + formatSide(self.b)
        return self.string

    def hasHole(self):

        if not self.hasHole2:
            return False

        self.hasHole2 = self.a.hasHole() or self.b.hasHole()

        return self.hasHole2

    def spread(self, case):
        self.string = None

        if type(self.a)==type((Hole())) and type(case)!=type(Question()):
            self.a = case
            return True
        elif type(self.b)==type((Hole())) and type(case)!=type(Question()) :
            if type(self.a) == type(Character('')) and type(case) == type(Character('')):
                if self.a.c == case.c:
                    return False
            self.b = case
            return True

        if self.a.spread(case):
            return True
        else:
            return self.b.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.a)==type((Hole())):
            self.a = KleenStar(Or(Character('0'), Character('1')))
        else:
            self.a.spreadAll()
        if type(self.b)==type((Hole())):
            self.b = KleenStar(Or(Character('0'), Character('1')))
        else:
            self.b.spreadAll()

    def spreadRand(self):
        self.string = None
        if type(self.a) == type((Hole())):
            self.a = Character('0') if random.random() <0.5 else Character('1')
        else:
            self.a.spreadRand()
        if type(self.b) == type((Hole())):
            self.a = Character('0') if random.random() <0.5 else Character('1')
        else:
            self.b.spreadRand()

    def spreadNp(self):
        self.string = None

        if type(self.a) == type((Hole())):
            self.a = Character('@emptyset')
        else:
            self.a.spreadNp()
        if type(self.b) == type((Hole())):
            self.b = Character('@emptyset')
        else:
            self.b.spreadNp()

    def unroll(self):
        self.string = None
        if type(self.a) == type(KleenStar()) and not self.a.hasHole():
            s = copy.deepcopy(self.a.r)
            self.a = Concatenate(Concatenate(s, s), KleenStar(s))

            self.a.a.unroll()
            self.a.b.unroll()

        else:
            self.a.unroll()

        if type(self.b) == type(KleenStar()) and not self.b.hasHole():
            s = copy.deepcopy(self.b.r)
            self.b = Concatenate(Concatenate(s, s), KleenStar(s))

            self.b.a.unroll()
            self.b.b.unroll()

        else:
            self.b.unroll()


    def split(self, side):
        '''self.string = None

        if type(self.a) == type(Or()):
            self.a = copy.deepcopy(self.a.a) if side == 0 else copy.deepcopy(self.a.b)
            return True
        elif type(self.a) == type(Question()):
            self.a = copy.deepcopy(self.a.r) if side == 0 else Epsilon()
            return True
        elif type(self.b) == type(Or()):
            self.a = copy.deepcopy(self.b.a) if side == 0 else copy.deepcopy(self.b.b)
            return True
        elif type(self.b) == type(Question()):
                return True


        if self.a.split(side):
            return True
        else:
            return self.b.split(side)'''
        return

    def make_child(self, count):
        if type(self.a) == type((Hole())):
            self.a = get_rand_re(count)
        else:
            self.a.make_child(count)

        if type(self.b) == type((Hole())):
            self.b = get_rand_re(count)
        else:
            self.b.make_child(count)



