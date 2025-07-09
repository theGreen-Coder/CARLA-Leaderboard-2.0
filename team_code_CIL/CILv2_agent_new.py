#!/usr/bin/env python

from datetime import datetime
import math
import os

import cv2
from leaderboard.autoagents import autonomous_agent
from leaderboard.autoagents.autonomous_agent import Track

import carla, torch
from scipy.optimize import fsolve

# Imports for CILv2_multiview_attention
from PIL import Image, ImageDraw, ImageFont
from CILv2_multiview.network.models_console import Models
import torchvision.transforms.functional as TF
from CILv2_multiview.configs import g_conf, merge_with_yaml, set_type_of_process
from CILv2_multiview.dataloaders.transforms import encode_directions_4, encode_directions_6, inverse_normalize, decode_directions_4, decode_directions_6
import numpy as np
import json
import pickle
from importlib import import_module


def get_entry_point():
    return 'CILpp_agent'

def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

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
        
        ########## LOADING MODEL ##########
        g_conf.immutable(False)
        merge_with_yaml("/home/yourname/Code/CARLA-Leaderboard-2.0/pretrained_models/CIL/CILv2.yaml", process_type='drive')
        # set_type_of_process('drive', root=os.environ["TRAINING_RESULTS_ROOT"])

        self._model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        # if torch.cuda.device_count() > 1 and g_conf.DATA_PARALLEL:
        #     print("Using multiple GPUs parallel! ")
        #     print(torch.cuda.device_count(), 'GPUs to be used: ', os.environ["CUDA_VISIBLE_DEVICES"])
        #     self._model = DataParallelWrapper(_model)
        checkpoint = torch.load("/home/yourname/Code/CARLA-Leaderboard-2.0/pretrained_models/CIL/CIL.pth")
        # print(self._model.name + '_' + str(checkpoint_number) + '.pth', "loaded from ",
        #         os.path.join(exp_dir, 'checkpoints'))
        # if isinstance(_model, torch.nn.DataParallel):
        #     self._model.module.load_state_dict(checkpoint['model'])
        # else:
        self._model.load_state_dict(checkpoint['model'])
        self._model.cuda()
        self._model.eval()
        
        self._model = self._model.to(self.device)


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
        # Record Video to see where we're going!
        # TO DO: Change Hard-Coded stuff
        central = input_data["rgb_central"][1]          
        cv2.imwrite(f'./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_{self.image_iteration:05d}.jpg', central)
        self.image_iteration += 1
        
        print(list(input_data.keys()))
        # print(input_data)
        
        # Run inputs through CILv2_multiview_attention
        self.control = carla.VehicleControl()
        
        print("Processing images")
        self.norm_rgb = [[self.process_image(input_data[camera_type][1]).unsqueeze(0).to(self.device) for camera_type in ["rgb_central", "rgb_left", "rgb_right"]]]

        print("Processing speed")
        self.norm_speed = [torch.cuda.FloatTensor([self.process_speed(input_data['SPEED'][1]['speed'])]).unsqueeze(0).to(self.device)]

        print("Processing direction")
        # self.direction = [torch.cuda.FloatTensor(self.process_command(input_data['GPS'][1], input_data['IMU'][1])[0]).unsqueeze(0).cuda()]
        self.direction = [torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32).unsqueeze(0).to(self.device)]

        actions_outputs, _, self.attn_weights = self._model.forward_eval(self.norm_rgb, self.direction, self.norm_speed)

        action_outputs = self.process_control_outputs(actions_outputs.detach().cpu().numpy().squeeze())

        self.steer, self.throttle, self.brake = action_outputs
        self.control.steer = float(self.steer)
        self.control.throttle = float(self.throttle)
        self.control.brake = float(self.brake)
        self.control.hand_brake = False
        
        # self.control.throttle = 1
        # self.control.brake    = 0.0
        # self.control.steer    = 0.0
        
        return self.control
    
    def process_image(self, image):
        image = Image.fromarray(image)
        image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])).convert('RGB')
        image = TF.to_tensor(image)
        
        # Normalization is really necessary if you want to use any pretrained weights.
        image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
        return image

    def process_speed(self, speed):
        norm_speed = abs(speed - g_conf.DATA_NORMALIZATION['speed'][0]) / (
                g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0])
        return norm_speed

    def process_control_outputs(self, action_outputs):
        if g_conf.ACCELERATION_AS_ACTION:
            steer, self.acceleration = action_outputs[0], action_outputs[1]
            if self.acceleration >= 0.0:
                throttle = self.acceleration
                brake = 0.0
            else:
                brake = np.abs(self.acceleration)
                throttle = 0.0
        else:
            steer, throttle, brake = action_outputs[0], action_outputs[1], action_outputs[2]
            if brake < 0.05:
                brake = 0.0

        return np.clip(steer, -1, 1), np.clip(throttle, 0, 1), np.clip(brake, 0, 1)

    def process_command(self, gps, imu):
        if g_conf.DATA_COMMAND_CLASS_NUM == 4:
            _, _, cmd = self.waypointer.tick_nc(gps, imu)
            return encode_directions_4(cmd.value), cmd.value
        elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
            _, _, cmd = self.waypointer.tick_lb(gps, imu)
            return encode_directions_6(cmd.value), cmd.value

    def destroy(self, results=None):
        print("Printing current folder!")
        os.system("pwd")
        print(f"ffmpeg -framerate 30 -i ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_%05d.jpg -c:v libx264 -pix_fmt yuv420p ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_color_video.mp4")
        os.system(f"ffmpeg -framerate 30 -i ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_%05d.jpg -c:v libx264 -pix_fmt yuv420p ./Bench2Drive/CIL_b2d_traj/central_{self.video_id}_color_video.mp4")
        os.system(f"rm ./Bench2Drive/CIL_b2d_traj/*.jpg")