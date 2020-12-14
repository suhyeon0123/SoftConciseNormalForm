import glob
import os
import fnmatch

class Examples(object):

    def __init__(self, no):
        '''self.benchmark = {0:[[],[],'null'],
                     1:[['0','00','001'],['1','11','101','100'],'start with 0, 0(0+1)*'],
                     2:[['01','001','1001'],['00','11','100','000','110','011'], 'end with 01, (0+1)*01'],
                     3:[['0101','001011','00010110'],['0','1','10','010','1110','1011','1011','1111','0010'], 'substring 0101, (0+1)*0101(0+1)*'],
                     4:[['10','110','1000', '11110'],['0','1','11','101','010','001'],'begin 1 end 0, 1(0+1)*0'],
                     5:[['110','1001','01001'],['1','01','101','0010'],'length at least 3 and third = 0, (0+1)(0+1)0(0+1)*'],
                     6:[['111','000','101','011','101101','110110','000111','101011'],['1','01','0','00','1111','1010','10110'],'lenis3mul, ((0+1)(0+1)(0+1))*']
                     }'''

        self.pos_simple, self.neg_simple, self.answer = self.readFromFile(no)
        self.pos = list()
        self.neg = list()
        self.make_examples(self.pos_simple,True)
        self.make_examples(self.neg_simple,False)

        '''self.readFromFiles()
        self.pos = self.benchmark[no][0]
        self.neg = self.benchmark[no][1]
        self.answer = self.benchmark[no][2]'''

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

    def readFromFiles(self):
        files = glob.glob('./benchmarks/*')
        for file in files:
            f = open(file, 'r') 
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
                
            self.benchmark[len(self.benchmark)] = [pos, neg, description.strip()]

    def  readFromFile(self, no):
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

        for i in simple :
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
