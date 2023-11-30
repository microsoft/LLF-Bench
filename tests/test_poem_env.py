import gymnasium as gym
import llfbench

# env = gym.make('llf-PickBestSummary-v0')

poem_env = gym.make('llf-poem-Haiku-v0', feedback=1)


poem = """
Beneath autumn trees,
Crimson leaves dance on the wind,
Whispers of goodbye.
As nature's gown turns to gold,
I'm left longing for the spring.
"""

assignment = poem_env.reset()
observation, reward, terminated, truncated, info = poem_env.step(poem)
done = terminated or truncated

print('Assignment: ', assignment, '\n', \
      'Observation: ', observation, '\n', \
      'Reward: ', reward, '\n', \
      'Done: ', done, '\n', \
      'Info: ', info)

poem_env = gym.make('llf-poem-SyllableConstrainedPoem-v0', feedback=1)
assignment = poem_env.reset()
observation, reward, terminated, truncated, info = poem_env.step(poem)
done = terminated or truncated

print('Assignment: ', assignment, '\n', \
      'Observation: ', observation, '\n', \
      'Reward: ', reward, '\n', \
      'Done: ', done, '\n', \
      'Info: ', info)
