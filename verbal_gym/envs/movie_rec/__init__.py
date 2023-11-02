from gymnasium.envs.registration import register

environments = [
    'MovieRec'
]

for environment in environments:
    register(
        id=f'verbal-{environment}-v0',
        entry_point=f'verbal_gym.envs.movie_rec.movie_rec:{environment}',
    )