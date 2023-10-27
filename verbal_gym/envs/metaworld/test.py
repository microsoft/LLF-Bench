import metaworld
import random



benchmark = metaworld.MT10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in benchmark.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in benchmark.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)
  breakpoint()