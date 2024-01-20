from gymnasium.envs.registration import register

register(
    id = 'my-merge-v0',
    entry_point = 'MyEnv.envs:MyMergeEnv')
    
register(
    id = 'my-highway-v0',
    entry_point = 'MyEnv.envs:MyHighwayEnv')
    
register(
    id = 'my-racetrack-v0',
    entry_point = 'MyEnv.envs:MyRacetrackEnv')
    
register(
    id = 'my-roundabout-v0',
    entry_point = 'MyEnv.envs:MyRoundaboutEnv')

register(
    id = 'test-racetrack-v0',
    entry_point = 'MyEnv.envs:testRacetrackEnv')
    
register(
    id = 'my-intersection-v0',
    entry_point = 'MyEnv.envs:MyIntersectionEnv')
    
register(
    id = 'my-crowded_highway-v0',
    entry_point = 'MyEnv.envs:MyCrowdedHighwayEnv')
    
register(
    id = 'my-crowded_merge-v0',
    entry_point = 'MyEnv.envs:MyCrowdedMergeEnv')
    
register(
    id = 'my-merge_hard-v0',
    entry_point = 'MyEnv.envs:MyMergeEnvHardReward')
    
register(
    id = 'my-crowded_merge_hard-v0',
    entry_point = 'MyEnv.envs:MyMergeEnvHardReward')
