import DQN

FIELD_SIZE = 40
STACKED_OBSERVATION = 30
GAMMA = 0.5

ITERATION = 200000
TEST_ITER = 2000
epsilon = 0.2

SHOW_ITERATION = 25
LR = 1e-5

env = DQN.environment(FIELD_SIZE)
agent = DQN.DQNAgent(env, FIELD_SIZE, STACKED_OBSERVATION, GAMMA, LR)
#agent = DQN.DQNAgent(env, FIELD_SIZE, STACKED_OBSERVATION, GAMMA)
#DQN.train(agent, ITERATION, epsilon, SHOW_ITERATION)
DQN.train(agent, ITERATION, epsilon, 300, True)

agent.save_model("./model_part.pt")
print("Now running with just greedy policy")
DQN.train(agent, TEST_ITER, 0, 100, True)

