# Global parameters (regex costs)

UNION_COST = 30
CONCAT_COST = 5
CLOSURE_COST = 20
SYMBOL_COST = 20
HOLE_COST = 100



# DQN parameters


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50000
TARGET_UPDATE = 1

LENGTH_LIMIT = 30
EXAMPLE_LENGTH_LIMIT = 100

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = 12

REPLAY_INITIAL = 10000
REPALY_MEMORY_SIZE = 1000000


# A2C parameters


NUM_PROCESSES = 1  # 동시 실행 환경 수
NUM_ADVANCED_STEP = 5 # 총 보상을 계산할 때 Advantage 학습을 할 단계 수
SHOW_ITER = 1

# A2C 손실함수 계산에 사용되는 상수
value_loss_coef = 0.5
entropy_coef = 0.01
policy_loss_coef = 1
max_grad_norm = 0.5

GAMMA = 0.9  # 시간할인율

lr = 1e-3
eps = 1e-5
alpha = 0.99