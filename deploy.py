#!/usr/bin/env python3
import random
import time
from your_own_robot_controller import RobotController
from utils import *
import pickle as pkl
import torch
import numpy as np
import copy
from torchvision.transforms import CenterCrop, Resize
from models import LSTMPolicy
import franka_controller.se3_tools as se3
from miles_dataset_loader import InteractionDataset
import matplotlib.pyplot as plt
import cv2


TASKS = ["key_insert_twist", "plug_in_socket", "usb_insertion", "insert_power_cable", "bread_in_toaster", "open_lid", "screwdiver", "bin_generalization"]
TASK_NAME_DEMO = TASKS[4]
ASSETS_DIR = "/path/to/assets/assets"
DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 150.0, 100.0, 20.0]
DEFAULT_K_GAINS_SAFETY = [30.0, 30.0, 30.0, 30.0, 12.0, 7.0, 2.0]
DAMPING_RATIO = 1.
WAIT_TILL_REPLAY = 20
RATE_AFTER_SEQ = 0.01

class Control:
    def __init__(self,
                 task_name='test',
                 data_collection_recording_rate=10,
                 action_horizon=6,
                 denormalize=True,
                 vision_only=False,
                 force_only=False):

        self.action_horizon = action_horizon
        self.denormalize = denormalize
        self.force_only = force_only
        self.vision_only = vision_only
        self.average_execution_time = 0

        # set up cuda
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        self.task_name = task_name

        self.robot = RobotController()
        self._recording_rate = data_collection_recording_rate


        demonstration_dir = "{}/tasks/{}/demonstration".format(ASSETS_DIR, task_name)
        demonstration_path = "{}/tasks/{}/demonstration/recorded_demo.pkl".format(ASSETS_DIR, task_name)
        starting_pose_path = "{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, task_name)

        self.demonstration = pkl.load(open(demonstration_path, 'rb'))
        self.starting_pose = pkl.load(open(starting_pose_path, 'rb'))
        self.demonstration_in_base = copy.deepcopy(self.convert_demo_trajectory_to_base_frame())
        data_folder = "closed_loop"

        self.train_dataset = InteractionDataset(
            path="{}/tasks/{}/data/{}/".format(ASSETS_DIR, self.task_name, data_folder),
            demonstration_path="{}/tasks/{}/demonstration".format(ASSETS_DIR, self.task_name),
            horizon=action_horizon,
            action_multiplier=6,
            subsample_frequency=1,
            mix_trajectories=False,
            normalize_actions=False)


        self.collected_data_details = torch.load("{}/tasks/{}/data/{}/data_collection_details.pt".format(ASSETS_DIR, task_name, data_folder))
        self.last_demo_index_collected =self.collected_data_details['demo_index']
        self.control_model = LSTMPolicy(action_dim=3 * self.action_horizon, with_gaussian_output=False,vision_only=self.vision_only, force_only=self.force_only,train_classifier=False).to(self.device)
        self.control_model_ori = LSTMPolicy(action_dim=3 * self.action_horizon, with_gaussian_output=False,vision_only=self.vision_only, force_only=self.force_only,train_classifier=False).to(self.device)
        extension = ""
        if self.vision_only:
            extension += "_vision_only"
        elif self.force_only:
            extension += "_force_only"
        lstm_extension=""
        self.control_model.load_state_dict(
            torch.load("{}/tasks/{}/models/closed_loop/policy_{}lstm_seq_{}_lin{}.pt".format(ASSETS_DIR, task_name, lstm_extension, action_horizon, extension)))
        self.control_model_ori.load_state_dict(
            torch.load("{}/tasks/{}/models/closed_loop/policy_{}lstm_seq_{}_ori{}.pt".format(ASSETS_DIR, task_name, lstm_extension, action_horizon, extension))) # if it can't find path it's because you may have changed something in this line
        
        norm_constants_lin = torch.load("{}/tasks/{}/data/{}/normalization_constants_lin_{}{}.pt".format(ASSETS_DIR, task_name, data_folder,action_horizon, extension))           
        self.min_norm_consts_lin, self.max_norm_consts_lin = norm_constants_lin['min_norm'], norm_constants_lin['max_norm']
        norm_constants_ang = torch.load("{}/tasks/{}/data/{}/normalization_constants_ang_{}{}.pt".format(ASSETS_DIR, task_name, data_folder,
                                                                                    action_horizon, extension))
        self.min_norm_consts_ang, self.max_norm_consts_ang = norm_constants_ang['min_norm'], norm_constants_ang['max_norm']
        self.control_model.eval()
        self.control_model_ori.eval()

    def denormalize_actions(self, data, denormalize_ori=False):

        """Denormalize the data in the range of [-1,+1] based on the min and max normalization constants."""
        if not self.denormalize:# or denormalize_ori:
            return data
        if denormalize_ori:
            min = self.min_norm_consts_ang
            max = self.max_norm_consts_ang

        else:
            # Denormalize every linear action separately
            min = self.min_norm_consts_lin
            max = self.max_norm_consts_lin
        data[0::3] = (data[0::3] + 1) * (max[0] - min[0]) / 2 + min[0]
        data[1::3] = (data[1::3] + 1) * (max[1] - min[1]) / 2 + min[1]
        data[2::3] = (data[2::3] + 1) * (max[2] - min[2]) / 2 + min[2]
        return data


    def convert_demo_trajectory_to_base_frame(self):
        demo_in_base = []
        for i, pose in enumerate(self.demonstration):
            demo_in_base.append(self.starting_pose @ pose)
        return demo_in_base

    def process_img(self, img):
        center_crop = CenterCrop(480)
        img_resize = Resize((128, 128))
        if img.shape[0] == img.shape[1]:
            img = img_resize(torch.tensor(img).permute(2, 0, 1) / 255)
        else:
            img = center_crop(torch.tensor(img).permute(2, 0, 1) / 255)
            img = img_resize(img)
        return img


    @torch.no_grad()
    def perform_task_closed_loop(self, idx=None,  auto_reset=False, load_poses_file=False):

        input("Press enter to start control")


        hidden_state = None
        hidden_state_ori = None
        submm_actions_predicted = 0
        consecutive_resets = 0
        stime = time.time()
        timesteps = 200
        for _ in range(timesteps):
            torch.cuda.empty_cache()
            current_eef_pose = copy.deepcopy(self.robot.get_eef_pose())
            current_img = self.camera.get_rgb(as_tensor=True).to(self.device)
            current_img = self.process_img(current_img)
            current_force = torch.from_numpy(self.robot.get_eef_wrench()).float().to(self.device)
            action_pred, hidden_state = self.control_model.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                   current_force.unsqueeze(0).unsqueeze(0),
                                                                   hidden_state)

            action_ori, hidden_state_ori = self.control_model_ori.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                               current_force.unsqueeze(0).unsqueeze(0),
                                                                               hidden_state_ori)

            action_pred[0, 0, :] = self.denormalize_actions(action_pred[0, 0, :3 * self.action_horizon])
            action_ori[0, 0, :3 * self.action_horizon] = self.denormalize_actions(
                action_ori[0, 0, :3 * self.action_horizon],
                denormalize_ori=True)

            action = torch.cat((action_pred, action_ori), dim=2)
            action = action.detach().cpu().numpy()[0]
            actions_in_eef = []
            for k in range(self.action_horizon):
                action_in_eef_0 = np.eye(4)
                action_in_eef_0[:3, 3] = action[0, k * 3:k * 3 + 3]
                action_in_eef_0[:3, :3] = se3.euler2rot("xyz", action[0,
                                                               3 * self.action_horizon + k * 3:3 * self.action_horizon + k * 3 + 3],
                                                        degrees=False)
                action_in_eef_0 = np.asarray(action_in_eef_0)
                actions_in_eef.append(action_in_eef_0)

            for _, action in enumerate(actions_in_eef):
                next_pose = current_eef_pose @ action
                """
                If you predict more than 15 submm actions, consider converged and switch to open-loop replay"""
                if np.abs(action[0, 3]) < 0.001 and np.abs(action[1, 3]) < 0.001 and np.abs(action[2, 3]) < 0.001: # if network converged switch to open-loop replay
                    submm_actions_predicted += 1
                elif submm_actions_predicted < 15:
                    submm_actions_predicted = 0
                self.robot.go_to_pose_asynchronously(next_pose)
                sleep_time = np.max((1 / self._recording_rate - (time.time() - stime), 0))
                current_img = self.robot.camera.get_rgb(as_tensor=True).to(self.device)
                current_force = torch.from_numpy(self.robot.get_eef_wrench()).float().to(self.device)
                current_img = self.process_img(current_img)
                force_felt = self.robot.get_eef_wrench()
                CONTROL_RATE_DEMO = 1
                time.sleep(CONTROL_RATE_DEMO/self._recording_rate)
                # (optional) check if the force is exceeded for safety
                if np.abs(force_felt[0]) > 50 \
                        or np.abs(force_felt[1]) > 50 \
                        or np.abs(force_felt[2]) > 50 \
                        or np.abs(force_felt[3]) > 20 \
                        or np.abs(force_felt[4]) > 20 \
                        or np.abs(force_felt[5]) > 20:
                    print("Force exceeded exiting experiment")
                    break
                    
                _, hidden_state = self.control_model.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                      current_force.unsqueeze(0).unsqueeze(0),
                                                                      hidden_state)
                _, hidden_state_ori = self.control_model.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                          current_force.unsqueeze(0).unsqueeze(0),
                                                                          hidden_state_ori)
            time.sleep(RATE_AFTER_SEQ)

            force_felt = self.robot.get_eef_wrench()
            if np.abs(force_felt[0]) > 50 \
                    or np.abs(force_felt[1]) > 50 \
                    or np.abs(force_felt[2]) > 50 \
                    or np.abs(force_felt[3]) > 20 \
                    or np.abs(force_felt[4]) > 20 \
                    or np.abs(force_felt[5]) > 20:
                print("Force exceeded exiting experiment")
                break

        self.robot.replay_demonstration(from_index=self.last_demo_index_collected)


if __name__ == '__main__':
    closed_loop_control = Control()
    closed_loop_control.perform_task()
