from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnvFast
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split
from highway_env.envs.common.action import Action

class MyHighwayEnvHardReward(HighwayEnvFast):
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
            "lanes_count": 2,
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0,
            "vehicles_count": 50,
            "distance": 1000,
            })
        return cfg
        
    def _reward(self, action: Action) -> float:
        if self.vehicle.position[0] > self.config["distance"]:
            return 1 / self.steps
        else:
            return 0
            
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.vehicle.position[0] > self.config["distance"]
            

        
    
    
