import llfbench

env = llfbench.make('llf-Alfworld-v0')

obs = env.reset()
for k, v in obs.items():
    if v is not None:
        print(f"Received observation with key={k}, value={v}")

for _ in range(5):
    action = input(f"enter action\n")
    obs, reward, terminated, truncated, info = env.step(action)
    for k, v in obs.items():
        if v is not None:
            print(f"Took action {action} and received observation with key={k}, value={v}")
