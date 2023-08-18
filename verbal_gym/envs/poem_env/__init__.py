from gym.envs.registration import register

environments = [
    'Haiku',
    'Tanka',
    'LineSyllableConstrainedPoem',
    'SyllableConstrainedPoem',
]

for environment in environments:
    register(
        id=f'verbal-{environment}-v0',
        entry_point=f'verbal_gym.envs.poem_env.formal_poems:{environment}',
    )
