import fnmatch
import os
from xeger import Xeger
#from parsetree import*
from parsetree_prune import*
from util import membership


class Examples(object):

    def __init__(self, benchmark, no = 1):
        self.benchmark = benchmark
        self.limit = 10
        self.pos = list()
        self.neg = list()

        if benchmark:
            self.pos_simple, self.neg_simple, self.answer = self.readFromFile(no)
            self.convertX(self.pos_simple, True)
            self.convertX(self.neg_simple, False)
        else:
            self.rand_example(self.limit)

    def rand_example(self, limit):
        while True:
            self.pos = list()
            self.neg = list()
            x = Xeger()
            regex = RE()
            for count in range(limit):
                regex.make_child(count)
            regex.spreadRand()
            regex = repr(regex)
            self.answer = regex

            pos_size = 8
            for i in range(1000):
                tmp = x.xeger(regex)
                if len(tmp) <= 15:
                    self.pos.append(tmp)
                    if len(self.pos) == pos_size:
                        break
            if not len(self.pos) == pos_size:
                continue

            neg_size = 8
            for i in range(1000):
                random_str = self.gen_str()
                if not membership(regex, random_str):
                    self.neg.append(random_str)
                    if len(self.neg) == neg_size:
                        break
            if not len(self.neg) == neg_size:
                continue

            break

    def gen_str(self):
        str_list = []

        for i in range(random.randrange(1, 10)):
            if random.randrange(1, 3) == 1:
                str_list.append('0')
            else:
                str_list.append('1')

        return ''.join(str_list)

    def getPos(self):
        return self.pos

    def getNeg(self):
        return self.neg

    def getAnswer(self):
        return self.answer

    def readFromFile(self, no):
        target_name = "no" + str(no) + "_*"
        for file_name in os.listdir("./benchmarks"):
            if fnmatch.fnmatch(file_name, target_name):
                f = open("./benchmarks/" + file_name, 'r')

        lines = f.readlines()
        description = ''
        index = 0
        pos = []
        neg = []

        while lines[index].strip() != '++':
            description += lines[index].strip() + ' '
            index += 1

        index += 1
        while lines[index].strip() != '--':
            pos.append(lines[index].strip())
            index += 1

        index += 1
        while index < len(lines):
            neg.append(lines[index].strip())
            index += 1

        return pos, neg, description.strip()

    def convertX(self, simple, is_pos):

        for i in simple:
            if 'X' in i:
                self.examples_rec(i.replace('X', '0', 1), is_pos)
                self.examples_rec(i.replace('X', '1', 1), is_pos)
            else:
                self.examples_rec(i.replace('X', '0', 1), is_pos)

    def examples_rec(self, i, is_pos):
        if 'X' in i:
            self.examples_rec(i.replace('X', '0', 1), is_pos)
            self.examples_rec(i.replace('X', '1', 1), is_pos)
        elif is_pos:
            self.pos.append(i)
        else:
            self.neg.append(i)

