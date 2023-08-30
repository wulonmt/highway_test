from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle

class MyRacetrackEnv(RacetrackEnv):
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
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 8, 16],
                        },
            "collision_reward": -5,
            "right_lane_reward": 0,
            "high_speed_reward": 0.4,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0,
            "screen_width": 1000,
            "screen_height": 1000,
            "other_vehicles": 4,
            "duration": 50,
            })
        return cfg

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            vehicle = Vehicle.create_random(
                self.road,
                speed=8,
                lane_from = lane_index[0],
                lane_to = lane_index[1],
                lane_id = lane_index[2],
            )
            controlled_vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=16+rng.normal()*2)
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(self.config["other_vehicles"]):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=16+rng.normal()*2)
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
        #for takeover
        self.closing = np.full((len(self.road.vehicles)-1), False)
                
    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        #reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1])
        reward *= rewards["on_road_reward"]
        #print("reward: ", reward, end='\r')
        self.takeover()
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle) / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
        }
        
    def takeover(self):
        ego = self.road.vehicles[0]
        others = self.road.vehicles[1:]
        vehicles_distance = []
        for i, other in enumerate(others):
            dist = np.linalg.norm(ego.position - other.position)
            if(dist < 10):
                print(f"{i} is closing")
        
