from gymnasium.envs.registration import register

register(
    id = 'my-merge-v0',
    entry_point = 'MyEnv.envs:MyMergeEnv')
