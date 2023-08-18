from gym.envs.registration import register

register(
    id='verbal-PickBestSummary-v0',
    entry_point='verbal_gym.envs.summarization.pick_best_summary:PickBestSummary',
)
