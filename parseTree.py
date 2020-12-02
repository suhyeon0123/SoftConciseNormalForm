#Regular Expression Implementation ,Written by Adrian Stoll
import sys
import copy

class Node:
    def __lt__(self, other):
        return False
    # All nodes have the following field and method:
    # level - stores prededence information to inform us of when to add parenthesis when pretty pritting
    # __repr__ - prints the regex represented by the parse tree with with any uneccearry parens removed
    pass

class Hole(Node):
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return '#'
    def hasHole(self):
        return True
    def spread(self):
        return
    def spreadAll(self):
        return copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
    def spreadNp(self):
        return Character('@emptyset')


class Epsilon(Node):
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

class EpsilonBlank(Node):
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

class Character(Node):
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

class KleenStar(Node):
    def __init__(self, r):
        self.r = r
        self.level = 1
    def __repr__(self):

        if '{}'.format(self.r) == '@emptyset':
            return '@epsilon'

        if '{}'.format(self.r) == '@epsilon':
            return '@epsilon'

        if self.r.level > self.level:
            return '({})*'.format(self.r)
        else:
            return '{}*'.format(self.r)

    def hasHole(self):
        return self.r.hasHole()

    def spread(self, case):
        if type(self.r)==type((Hole())):
            if type(case) != type(KleenStar(Hole())):
                self.r = case
                return True
            else:
                return False

        return self.r.spread(case)

    def spreadAll(self):
        if type(self.r) == type((Hole())):
            self.r = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.r.spreadAll()

    def spreadNp(self):
        if type(self.r) == type((Hole())):
            self.r = Character('@emptyset')
        else:
            self.r.spreadNp()


class Concatenate(Node):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.level = 2
    def __repr__(self):

        def formatSide(side):
            if side.level > self.level:
                return '({})'.format(side)
            else:
                return '{}'.format(side)

        if '@emptyset' in repr(self.a) or '@emptyset' in repr(self.b):
            return '@emptyset'

        if '@epsilon' == repr(self.a):
            return formatSide(self.b)
        elif '@epsilon' == repr(self.b):
            return formatSide(self.a)

        return formatSide(self.a) + formatSide(self.b)

    def hasHole(self):
        return self.a.hasHole() or self.b.hasHole()

    def spread(self, case):
        if type(self.a)==type((Hole())):
            self.a = copy.deepcopy(case)
            return True
        elif type(self.b)==type((Hole())):
            self.b = copy.deepcopy(case)
            return True
        if self.a.spread(case):
            return True
        else:
            return self.b.spread(case)

    def spreadAll(self):
        if type(self.a)==type((Hole())):
            self.a = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.a.spreadAll()
        if type(self.b)==type((Hole())):
            self.b = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))

        else:
            self.b.spreadAll()

    def spreadNp(self):
        if type(self.a)==type((Hole())):
            self.a = Character('@emptyset')
        else:
            self.a.spreadNp()

        if type(self.b)==type((Hole())):
            self.b = Character('@emptyset')
        else:
            self.b.spreadNp()

class Or(Node):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.level = 3

    def __repr__(self):

        def formatSide(side):
            if side.level > self.level:
                return '({})'.format(side)
            else:
                return '{}'.format(side)

        if repr(self.a) == '@emptyset':
            return formatSide(self.b)
        elif repr(self.b) == '@emptyset':
            return formatSide(self.a)

        return formatSide(self.a) + '+' + formatSide(self.b)

    def hasHole(self):
        return self.a.hasHole() or self.b.hasHole()

    def spread(self, case):
        if type(self.a)==type((Hole())):
            self.a = copy.deepcopy(case)
            return True
        elif type(self.b)==type((Hole())):
            self.b = copy.deepcopy(case)
            return True
        if self.a.spread(case):
            return True
        else:
            return self.b.spread(case)

    def spreadAll(self):
        if type(self.a)==type((Hole())):
            self.a = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.a.spreadAll()
        if type(self.b)==type((Hole())):
            self.b = copy.deepcopy(KleenStar(Or(Character('0'), Character('1'))))
        else:
            self.b.spreadAll()

    def spreadNp(self):
        if type(self.a) == type((Hole())):
            self.a = Character('@emptyset')
        else:
            self.a.spreadNp()
        if type(self.b) == type((Hole())):
            self.b = Character('@emptyset')
        else:
            self.b.spreadNp()
