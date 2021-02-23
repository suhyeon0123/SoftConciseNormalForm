from parsetree_prune import*
from xeger import Xeger
import re2 as re

limit = 5





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


for no in range(1, 200):
    regex = rand_example(limit)

    x = Xeger()
    posset = set()
    for i in range(0, 15):
        posset.add(x.xeger(repr(regex)))

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

        if len(negset) == 15:
            break

    if not negset:
        continue

    fname = "rand_benchmarks/no" + str(no) + ".txt"
    f = open(fname, 'wt')

    f.write(repr(regex)+"\n")

    f.write('++\n')

    for index, i in enumerate(posset):
        if i!= '' : f.write(i+"\n")

    f.write('--\n')

    for index, i in enumerate(negset):
        if i!= '' : f.write(i)

    f.close()

