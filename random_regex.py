from parsetree_prune import*
from xeger import Xeger
import re2 as re

limit = 10





def rand_example(limit):
    regex = RE()
    for count in range(limit):
        regex.make_child(1)
    regex.spreadRand()
    return regex

'''regex = rand_example(limit)
print(regex)
x = Xeger()
posset = set()
for i in range(0,10):
    posset.add(x.xeger(repr(regex)))
for index, i in enumerate(posset):
    print(i)
exit()'''

no = 0
while no<1000:
    regex = rand_example(limit)

    if len(repr(regex)) < 7:
        continue
    x = Xeger()


    posset = set()
    endcount = 0
    while endcount <50 and len(posset)<10 :
        posset.add(x.xeger(repr(regex)))
        endcount +=1



    negset = set()
    for i in range(0, 1000):
        # random regex생성
        str_list = []

        for j in range(0, random.randrange(1, 15)):
            if random.random() < 0.5:
                str_list.append('0')
            else:
                str_list.append('1')
        tmp = ''.join(str_list)

        # random regex가 맞지 않다면 추가
        if not bool(re.fullmatch(repr(regex), tmp)):
            negset.add(tmp + "\n")

        if len(negset) == 10:
            break

    if not len(negset) == 10:
        continue



    fname = "rand3_benchmarks/no" + str(no) + ".txt"
    f = open(fname, 'w')

    f.write(repr(regex)+"\n")

    f.write('++\n')

    for index, i in enumerate(posset):
        if i!= '' : f.write(i+"\n")

    f.write('--\n')

    for index, i in enumerate(negset):
        if i!= '' : f.write(i)

    f.close()
    print(no)
    no +=1

