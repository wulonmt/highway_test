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
