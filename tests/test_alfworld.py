import llfbench

env = llfbench.make('llf-Alfworld-v0')

obs = env.reset(seed=1234)
print("Observation is ", obs)

for _ in range(5):
    action = input(f"enter action\n")
    obs, reward, terminated, truncated, info = env.step(action)
    print("observation is ", obs)