#Regular Expression Implementation ,Written by Adrian Stoll
import sys
import copy

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
        return copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
    def spreadNp(self):
        return Character('@emptyset')
    def unroll(self):
        return
    def split(self):
        return



class RE:
    def __lt__(self, other):
        return False
    def __init__(self, r=Hole()):
        self.r = r
        self.hasHole2 = True
        self.string = None
        self.first = True

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
        self.string = None

        if self.first:
            self.r = copy.deepcopy(case)
            self.first = False
            return True
        else:
            return self.r.spread(case)
    def spreadAll(self):
        self.string = None
        self.r.spreadAll()
    def spreadNp(self):
        self.string = None
        self.r.spreadNp()
    def unroll(self):
        return
    def split(self):
        return


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
    def spreadNp(self):
        return
    def unroll(self):
        return
    def split(self):
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
    def spreadNp(self):
        return
    def unroll(self):
        return
    def split(self):
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
    def spreadNp(self):
        return self.c
    def unroll(self):
        return
    def split(self):
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
            if type(case) != type(KleenStar()) and type(case) != type(Question()):
                self.r = case
                return True
            else:
                return False

        return self.r.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.r) == type((Hole())):
            self.r = copy.deepcopy(Or(Character('0'), Character('1')))
        else:
            self.r.spreadAll()

    def spreadNp(self):
        self.string = None

        if type(self.r) == type((Hole())):
            self.r = Character('@emptyset')
        else:
            self.r.spreadNp()

    def unroll(self):
        return
    def split(self):
        return


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
            if type(case) != type(KleenStar()) and type(case) != type(Question()):
                self.r = case
                return True
            else:
                return False

        return self.r.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.r) == type(Hole()):
            self.r = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))

        else:
            self.r.spreadAll()

    def spreadNp(self):
        self.string = None

        if type(self.r) == type(Hole()):
            self.r = Character('@emptyset')
        else:
            self.r.spreadNp()

    def unroll(self):
        return
    def split(self):
        return


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

        if self.a.hasHole():
            if type(self.a) == type(Hole()):
                if type(case) != type(Concatenate()):
                    self.a = copy.deepcopy(case)
                    return True
                else:
                    return False
            else:
                return self.a.spread(case)
        elif self.b.hasHole():
            if type(self.b) == type(Hole()):
                self.b = copy.deepcopy(case)
                return True
            else:
                return self.b.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.a)==type(Hole()):
            self.a = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.a.spreadAll()
        if type(self.b)==type(Hole()):
            self.b = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))

        else:
            self.b.spreadAll()

    def spreadNp(self):
        self.string = None

        if type(self.a)==type(Hole()):
            self.a = Character('@emptyset')
        else:
            self.a.spreadNp()

        if type(self.b)==type(Hole()):
            self.b = Character('@emptyset')
        else:
            self.b.spreadNp()

    def unroll(self):
        return
    def split(self):
        return


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

        if self.a.hasHole():
            if type(self.a) == type(Hole()):
                if type(case) != type(Or()):
                    self.a = copy.deepcopy(case)
                    return True
                else:
                    return False
            else:
                return self.a.spread(case)
        elif self.b.hasHole():
            if type(self.b) == type(Hole()):
                if type(self.a) == type(Character('')) and type(case) == type(Character('')):
                    if self.a.c == case.c:
                        return False
                else:
                    self.b = copy.deepcopy(case)
                    return True                    
                return False
            else:
                return self.b.spread(case)

    def spreadAll(self):
        self.string = None

        if type(self.a)==type(Hole()):
            self.a = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.a.spreadAll()
        if type(self.b)==type(Hole()):
            self.b = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.b.spreadAll()

    def spreadNp(self):
        self.string = None

        if type(self.a) == type(Hole()):
            self.a = Character('@emptyset')
        else:
            self.a.spreadNp()
        if type(self.b) == type(Hole()):
            self.b = Character('@emptyset')
        else:
            self.b.spreadNp()

    def unroll(self):
        return
    def split(self):
        return


