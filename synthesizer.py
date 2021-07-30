from queue import PriorityQueue
from util import *
import copy


def synthesis(examples, count_limit=50000):
    w = PriorityQueue()
    scanned = set()
    w.put((REGEX().getCost(), REGEX()))

    i = 0
    traversed = 1

    answer = None
    finished = False

    while not w.empty() and not finished and i < count_limit:
        tmp = w.get()
        s = tmp[1]
        cost = tmp[0]

        hasHole = s.hasHole()

        print("state : ", s, " cost: ", cost)
        if hasHole:
            for j, new_elem in enumerate([Character('0'), Character('1'), Or(), Or(Character('0'), Character('1')),
                                          Concatenate(Hole(), Hole()), KleenStar(), Question()]):

                k = copy.deepcopy(s)

                if not k.spread(new_elem):
                    continue

                traversed += 1
                if repr(k) in scanned:
                    continue
                else:
                    scanned.add(repr(k))

                if not k.hasHole():
                    if is_solution(repr(k), examples, membership):
                        answer = repr(k)
                        finished = True
                        break

                if repr(new_elem) == '0|1' or new_elem.type == Type.CHAR:
                    checker = True
                else:
                    checker = False

                if checker and is_pdead(k, examples):
                    continue

                if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and is_ndead(k, examples):
                    continue

                if is_not_scnf(k, new_elem):
                    continue

                if is_redundant(k, examples, new_elem):
                    continue

                w.put((k.getCost(), k))
        i = i + 1

    return answer
