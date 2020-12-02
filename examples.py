class Examples(object):

    def __init__(self, no):
        venchmark = {0:[[],[],'null'],
                     1:[['0','00','001'],['1','11','101','100'],'start with 0, 0(0+1)*'],
                     2:[['01','001','1001'],['00','11','100','000','110','011'], 'end with 01, (0+1)*01'],
                     3:[['0101','001011','00010110'],['0','1','10','010','1110','1011','1011','1111','0010'], 'substring 0101, (0+1)*0101(0+1)*'],
                     4:[['10','110','1000', '11110'],['0','1','11','101','010','001'],'begin 1 end 0, 1(0+1)*0'],
                     5:[['110','1001','01001'],['1','01','101','0010'],'length at least 3 and third = 0, (0+1)(0+1)0(0+1)*'],
                     6:[['111','000','101','011','101101','110110','000111','101011'],['1','01','0','00','1111','1010','10110'],'lenis3mul, ((0+1)(0+1)(0+1))*']
                     }

        self.pos = venchmark[no][0]
        self.neg = venchmark[no][1]
        self.answer = venchmark[no][2]

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
