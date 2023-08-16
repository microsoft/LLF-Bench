import pdb
from envs.summarization.pick_best_summary import PickBestSummary


num_actions = 4
env = PickBestSummary(num_actions=num_actions, fixed=True, reward_type="binary")

for i in range(10):

    observation = env.reset()
    print(f"{i} Observation {observation}")

    action = input(f"Enter the best summary in {{0, 1, 2, .., {num_actions - 1}}}")
    action = int(action)
    next_observation, reward, done, info = env.step(action)

    print(f"{i} Reward {reward}, Done {done}, feedback {info['feedback']}")
    pdb.set_trace()
