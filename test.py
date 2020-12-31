from queue import PriorityQueue
import time
import configparser
from util import *
import argparse
from DQN import*

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", type=int,
                    help="Example number")
parser.add_argument("-u", "--unambiguous", help="Set ambiguity",
                    action="store_true")
parser.add_argument("-r", "--redundant", help="Set redundancy checker", action="store_true")
args = parser.parse_args()


#-----------------------------------


def select_action(regex_tensor, pos_tensor, neg_tensor):
    with torch.no_grad():
        #print(regex_tensor, pos_tensor)
        a = policy_net(regex_tensor.view(1,-1), pos_tensor.view(1,-1), neg_tensor.view(1,-1)) #(1,6)
        print(tensor_to_regex(regex_tensor.view(1,-1)), "\t\t", torch.argmax(a).view(-1,1)[0][0].item())
        #return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return torch.argmax(a).view(-1,1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 50000
TARGET_UPDATE = 1

LENGTH_LIMIT = 30
EXAMPLE_LENGHT_LIMIT = 100
REPLAY_INITIAL = 10000
REPALY_MEMORY_SIZE = 1000000

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = 6
embed_n = 500

policy_net = DQN().to(device)

policy_net.load_state_dict(torch.load('saved_model/DQN.pth'))
#policy_net.load_state_dict(torch.load('saved_model/PrioritizedDQN.pth'))
policy_net.eval()

sys.setrecursionlimit(5000000)

import faulthandler

faulthandler.enable()

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']



w = PriorityQueue()

scanned = set()

w.put((RE().cost, RE()))
examples = Examples(args.examples)
answer = examples.getAnswer()

print(examples.getPos(), examples.getNeg())

i = 0
traversed = 1
start = time.time()
prevCost = 0

success = False

while not w.empty() and not success:

    tmp = w.get()
    state = tmp[1]
    cost = tmp[0]

    prevCost = cost

    if not state.hasHole():
        continue

    for t in range(5):
        chosen_action = select_action(*make_embeded(state, examples))

        buffer = cost

        for j, new_elem in enumerate(
                [Character('0'), Character('1'), Or(), Concatenate(), KleenStar(), Question()]):

            k = copy.deepcopy(state)

            if not k.spread(new_elem):
                continue

            if len(repr(k)) > LENGTH_LIMIT:
                continue

            traversed += 1
            if repr(k) in scanned:
                # print("Already scanned?", repr(k))
                # print(list(scanned))
                continue
            else:
                scanned.add(repr(k))

            if is_pdead(k, examples):
                # print(repr(k), "is pdead")
                continue

            if is_ndead(k, examples):
                # print(repr(k), "is ndead")
                continue

            if args.redundant:
                if is_redundant(k, examples):
                    # print(repr(k), "is redundant")
                    continue

            if not k.hasHole():
                if is_solution(repr(k), examples, membership):
                    end = time.time()
                    print("Spent computation time:", end - start)
                    print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:",
                          w.qsize(), "\tTraversed:", traversed)
                    # print("Result RE:", repr(k), "Verified by FAdo:", is_solution(repr(k), examples, membership2))
                    print("Result RE:", repr(k))
                    success = True
                    break

            if j != chosen_action[0][0].item():
                w.put((k.cost, k))
            else:
                buffer = k.cost

        if success:
            break
        else:
            next_state, reward, done, success = make_next_state(state, chosen_action, examples)

        cost = buffer
        state = next_state

        if i % 1000 == 0:
            print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned),
                  "\tQueue Size:", w.qsize())

        i = i + 1

        if done:
            break

print('Complete')




