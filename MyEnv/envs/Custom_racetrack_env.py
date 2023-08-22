from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle

class CustomRacetrackEnv(RacetrackEnv):
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
            "screen_width": 1000,
            "screen_height": 1000,
            })
        return cfg
        
    def _make_road(self) -> None:
        net = RoadNetwork()
        width = 10

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane, the straight line below
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=width, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([42, width], [100, width], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=width, speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=width,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+width, np.deg2rad(90), np.deg2rad(-1), width=width,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=width,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([120+width, -20], [120+width, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=width,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=width,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+width, np.deg2rad(0), np.deg2rad(-181), width=width,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3+width, np.deg2rad(0), np.deg2rad(136), width=width,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=width,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=width,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=width,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=width,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4+width, np.deg2rad(315), np.deg2rad(165), width=width,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=width,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4+width, np.deg2rad(170), np.deg2rad(58), width=width,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5+width, np.deg2rad(240), np.deg2rad(270), width=width,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=width,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

