from parsetree import*

limit = 10

'''regex = RE(Concatenate(KleenStar(Character('0')),Hole()))
print(regex)
regex.unroll()
print(regex)'''




def rand_example(limit):
    regex = RE()
    for count in range(limit):
        regex.make_child()
    regex.spreadRand()
    return regex

print(rand_example(limit))

