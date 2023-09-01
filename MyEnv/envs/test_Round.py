from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle
import time

class testRacetrackEnv(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {"observation": {
                       "type": "GrayscaleObservation",
                       "observation_shape": (128, 64),
                       "stack_size": 4,
                       "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                       "scaling": 1.75,
                   },
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0,
            "screen_width": 1300,
            "screen_height": 600,
            })
        return cfg
        
    def _make_road(self) -> None:
        net = RoadNetwork()
        width = 20
        radii = 30

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]
        
        center2 = np.array([50, -radii])
        #度數的前減後要大於0
        #clockwise = false -> 90度在下面
        net.add_lane("d", "a",
                     CircularLane(center2, radii, np.deg2rad(-90), np.deg2rad(-270), width=width,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))
        
        # Initialise First Lane, the straight line below
        lane = StraightLane([50, 0], [150, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), width=width, speed_limit=speedlimits[1])
        self.lane = lane
        
        net.add_lane("a", "b", lane)

        # 2 - Circular Arc #1
        center1 = [150, 0 - radii]
        net.add_lane("b", "c",
                     CircularLane(center1, radii, np.deg2rad(90), np.deg2rad(-90), width=width,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))
        net.add_lane("c", "d",
                     StraightLane([150, -2*radii], [50, -2*radii], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), width=width, speed_limit=speedlimits[1]))
        
        

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", 0)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", 0)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(50, 100))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", 0),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)