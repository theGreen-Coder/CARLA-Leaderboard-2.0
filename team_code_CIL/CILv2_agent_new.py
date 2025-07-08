#!/usr/bin/env python

from datetime import datetime
import math
import os

import cv2
from leaderboard.autoagents import autonomous_agent
from leaderboard.autoagents.autonomous_agent import Track
import carla, torch
from scipy.optimize import fsolve

def get_entry_point():
    return 'CILpp_agent'

class CILpp_agent(autonomous_agent.AutonomousAgent):
    """
    CIL++ agent - minimal, ready to expand
    """

    # keep base-class constructor untouched
    def __init__(self, *args, **kwargs):
        print("THIS ACTUALLY WORKING INIT!")
        self.writer = None
        self.video_id = str(datetime.now().strftime("%m%d%H%M%S"))
        super().__init__(*args, **kwargs)

    def setup(self, path_to_conf_file):
        print("THIS ACTUALLY WORKING SETUP!")
        print(f"PATH to CONFIG: {str(path_to_conf_file)}")
        self.track   = Track.SENSORS
        self.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        self.initialized = False
        self.image_iteration = 0
        # TODO: load config + model here

    def sensors(self):
        # Common pose for all “virtual” sensors
        default_pose = dict(x=0.0, y=0.0, z=2.0,
                            roll=0.0, pitch=0.0, yaw=0.0)

        return [
            {**default_pose,
            'type': 'sensor.camera.rgb', 'id': 'rgb_central',
            'width': 300, 'height': 300, 'fov': 60,
            'lens_circle_setting': False},

            {**default_pose, 'yaw': -60.0,
            'type': 'sensor.camera.rgb', 'id': 'rgb_left',
            'width': 300, 'height': 300, 'fov': 60,
            'lens_circle_setting': False},

            {**default_pose, 'yaw': 60.0,
            'type': 'sensor.camera.rgb', 'id': 'rgb_right',
            'width': 300, 'height': 300, 'fov': 60,
            'lens_circle_setting': False},

            # {**default_pose,
            # 'type': 'sensor.other.gnss', 'id': 'GPS'},

            {**default_pose,
            'type': 'sensor.other.imu', 'id': 'IMU'},

            {**default_pose,
            'type': 'sensor.speedometer', 'id': 'SPEED'},
        ]

    def run_step(self, input_data, timestamp):
        print("THIS ACTUALLY WORKING RUN_STEP!")
        """
        Build your perception + control logic here
        """
        central = input_data["rgb_central"][1]
                                                      
        cv2.imwrite(f'./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_{self.image_iteration:05d}.jpg', central)
        
        self.image_iteration += 1

        control = carla.VehicleControl()
        control.throttle = 1
        control.brake    = 0.0
        control.steer    = 0.0
        
        return control

    def destroy(self, results=None):
        print("Printing current folder!")
        os.system("pwd")
        print(f"ffmpeg -framerate 30 -i ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_%05d.jpg -c:v libx264 -pix_fmt yuv420p ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_color_video.mp4")
        os.system(f"ffmpeg -framerate 30 -i ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_%05d.jpg -c:v libx264 -pix_fmt yuv420p ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_color_video.mp4")
        os.system(f"rm ./Bench2Drive/CIL_b2d_traj/*.jpg")