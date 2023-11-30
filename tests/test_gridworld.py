import llfbench

env = llfbench.make('llf-Gridworld-v0')

obs = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    for k, v in obs.items():
        print(f"Took action {action} and received observation with key={k}, value={v}")
