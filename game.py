from queue import PriorityQueue
from util import *
from models import *
from parsetree import *
from examples import *


class Game:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.done = False
        self.w = PriorityQueue()
        self.scanned = set()
        self.success = False
        self.examples = Examples(2)

        self.current_state = RE()
        self.start_time = time.time()

        self.iterations = 0

    def state(self):
        return self.current_state

    def step(self, action):

        self.iterations += 1

        use_queue = action % 2 == 0
        chosen_action = action // 2

        for j, new_elem in enumerate(
                [Character('0'), Character('1'), Or(), Concatenate(), KleenStar(), Question()]):

            k = copy.deepcopy(self.current_state)

            if not k.spread(new_elem) or len(repr(k)) > LENGTH_LIMIT:
                self.done = True
                continue

            if repr(k) in self.scanned:
                self.done = True
                continue
            else:
                self.scanned.add(repr(k))

            if is_pdead(k, self.examples):
                self.done = True
                continue

            if is_ndead(k, self.examples):
                self.done = True
                continue

            # if args.redundant:
            #    if is_redundant(k, examples):
            #        # print(repr(k), "is redundant")
            #        continue

            if not k.hasHole():
                if is_solution(repr(k), self.examples, membership):
                    if self.verbose:
                        print("Spent computation time:", time.time() - self.start_time)
                        print("Iteration:", self.iterations, "\tCost:", self.current_state.cost, "\tScanned REs:",
                              len(self.scanned), "\tQueue Size:",
                              self.w.qsize())
                        # print("Result RE:", repr(k), "Verified by FAdo:", is_solution(repr(k), examples, membership2))
                        print("Result RE:", repr(k))
                    success = True

                    return k, 100, success, j * 2 + use_queue

            if j != chosen_action:
                self.w.put((k.cost, k))

        self.current_state, reward, self.done, success = make_next_state(self.current_state, chosen_action, self.examples)

        if self.iterations % 100 == 0:
            if self.verbose:
                print("Iteration:", self.iterations, "\tCost:", self.current_state.cost, "\tScanned REs:",
                      len(self.scanned),
                      "\tQueue Size:", self.w.qsize())

        if use_queue and not self.done:
            self.done = True

        if self.done:
            self.w.put((self.current_state.cost, self.current_state))
            tmp = self.w.get()
            self.current_state = tmp[1]

        return self.current_state, reward, success, action
