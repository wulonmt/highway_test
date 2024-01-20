from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnvFast, HighwayEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle

class MyCrowdedHighwayEnv(HighwayEnvFast):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {"observation": {
                       "type": "GrayscaleObservation",
                       "observation_shape": (84, 84),
                       "stack_size": 4,
                       "weights": [0.9, 0.1, 0.5],  # weights for RGB conversion
                       "scaling": 1.75,
                   },
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0,
            "vehicles_density": 2,
            })
        return cfg
        
    
    
