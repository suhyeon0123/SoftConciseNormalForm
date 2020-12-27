import fnmatch
import glob
import os


class Examples(object):
    def __init__(self, no):
        self.pos_simple, self.neg_simple, self.answer = self.readFromFile(no)
        self.pos = list()
        self.neg = list()
        self.make_examples(self.pos_simple, True)
        self.make_examples(self.neg_simple, False)


    def setPos(self, pos):
        self.pos = pos

    def setNeg(self, neg):
        self.neg = neg

    def addPos(self, example):
        self.pos.append(example)

    def addNeg(self, example):
        self.neg.append(example)

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

    def make_examples(self, simple, is_pos):

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