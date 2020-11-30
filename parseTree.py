#Regular Expression Implementation ,Written by Adrian Stoll
import sys
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
    def spread(self, case):
        return True
    def spreadAll(self, case):
        return
    def spreadAll(self, case):
        return


class Epsilon(Node):
    def __init__(self):
        self.level = 0
    def __repr__(self):
        return '@epsilon'
    def hasHole(self):
        return False
    def spread(self, case):
        return False
    def spreadAll(self, case):
        return
    def spreadAll(self, case):
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
    def spreadAll(self, case):
        return
    def spreadAll(self, case):
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
    def spreadAll(self, case):
        return
    def spreadNp(self):
        return

class KleenStar(Node):
    def __init__(self, r):
        self.r = r
        self.level = 1
    def __repr__(self):
        if self.r.level > self.level:
            return '({})*'.format(self.r)
        else:
            return '{}*'.format(self.r)
    def hasHole(self):
        return self.r.hasHole()
    def spread(self, case):
        if type(self.r)==type((Hole())):
            self.r = case
            return True
        if self.r.spread(case):
            True
        else:
            False
    def spreadAll(self, case):
        if type(self.r) == type((Hole())):
            self.r = case
        else:
            self.r.spreadAll(case)
    def spreadNp(self):
        if type(self.r) == type((Hole())):
            self.r = Epsilon()
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
        return formatSide(self.a) + formatSide(self.b)
    def hasHole(self):
        return self.a.hasHole() or self.b.hasHole()
    def spread(self, case):
        if type(self.a)==type((Hole())):
            self.a = case
            return True
        elif type(self.b)==type((Hole())):
            self.b = case
            return True
        if self.a.spread(case):
            True
        else:
            self.b.spread(case)
    def spreadAll(self, case):
        if type(self.a)==type((Hole())):
            self.a = case
        else:
            self.a.spreadAll(case)
        if type(self.b)==type((Hole())):
            self.b = case
        else:
            self.b.spreadAll(case)
    def spreadNp(self):
        if type(self.a)==type((Hole())):
            self.a = EpsilonBlank()
        else:
            self.a.spreadNp()
        if type(self.b)==type((Hole())):
            self.b = EpsilonBlank()
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
        return formatSide(self.a) + '+' + formatSide(self.b)
    def hasHole(self):
        return self.a.hasHole() or self.b.hasHole()
    def spread(self, case):
        if type(self.a)==type((Hole())):
            self.a = case
            return True
        elif type(self.b)==type((Hole())):
            self.b = case
            return True
        if self.a.spread(case):
            True
        else:
            self.b.spread(case)
    def spreadAll(self, case):
        if type(self.a)==type((Hole())):
            self.a = case
        else:
            self.a.spreadAll(case)
        if type(self.b)==type((Hole())):
            self.b = case
        else:
            self.b.spreadAll(case)

    def spreadNp(self):
        if type(self.a) == type((Hole())):
            self.a = Epsilon()
        else:
            self.a.spreadNp()
        if type(self.b) == type((Hole())):
            self.b = Epsilon()
        else:
            self.b.spread()
