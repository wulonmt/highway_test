from gymnasium.envs.registration import register

register(
    id = 'my-merge-v0',
    entry_point = 'MyEnv.envs:MyMergeEnv')
    
register(
    id = 'my-highway-v0',
    entry_point = 'MyEnv.envs:MyHighwayEnv')
    
register(
    id = 'custom-racetrack-v0',
    entry_point = 'MyEnv.envs:CustomRacetrackEnv')
    
register(
    id = 'my-roundabout-v0',
    entry_point = 'MyEnv.envs:MyRoundaboutEnv')

    id = 'test-racetrack-v0',
    entry_point = 'MyEnv.envs:testRacetrackEnv')
