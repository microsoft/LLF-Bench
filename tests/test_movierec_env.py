import gymnasium as gym
import llfbench

action = a = """[
  {"title": "John Wick", "year": "2014", "platform": "Netflix", "genre": "action"},
  {"title": "Mad Max: Fury Road", "year": "2015", "platform": "Netflix", "genre": "action"},
  {"title": "Baby Driver", "year": "2017", "platform": "Netflix", "genre": "action"},
  {"title": "Avengers: Infinity War", "year": "2018", "platform": "Netflix", "genre": "action"},
  {"title": "Mission: Impossible - Fallout", "year": "2018", "platform": "Hulu/HBO Max", "genre": "action"},
  {"title": "Extraction", "year": "2020", "platform": "Netflix", "genre": "action"},
  {"title": "Wonder Woman", "year": "2017", "platform": "Netflix", "genre": "action"},
  {"title": "The Raid: Redemption", "year": "2011", "platform": "YouTube", "genre": "action"},
  {"title": "The Dark Knight", "year": "2008", "platform": "Netflix", "genre": "action"},
  {"title": "The Old Guard", "year": "2020", "platform": "Netflix", "genre": "action"}
]"""

movie_env = gym.make('verbal-MovieRec-b-hp-v0')

assignment = movie_env.reset()
observation, reward, done, truncated, info = movie_env.step(action)

print('Assignment: ', assignment, '\n', \
      'Observation: ', observation, '\n', \
      'Reward: ', reward, '\n', \
      'Done: ', done, '\n', \
      'Info: ', info)
