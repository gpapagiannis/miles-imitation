#!/usr/bin/env python3
import time
from your_own_robot_controller import RobotController
from utils import *
import pickle as pkl
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from torchvision.transforms import CenterCrop, Resize
from utils import bcolors
from dino_vit_features import find_sim
import matplotlib.pyplot as plt
import cv2
ASSETS_DIR = "/path/to/assets"
SIMILARITY_THRESHOLD = .94

class DataCollector:
    def __init__(self, task_name='test', data_collection_recording_rate=10, number_of_samples=10000):

        self.task_name = task_name

        make_dir(os.path.join(ASSETS_DIR, 'tasks'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'data'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'data', 'miles'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'demonstration'))

        self.robot = RobotController()
        self._recording_rate = data_collection_recording_rate


        demonstration_dir = "{}/tasks/{}/demonstration".format(ASSETS_DIR, task_name)
        demonstration_path = "{}/tasks/{}/demonstration/recorded_demo.pkl".format(ASSETS_DIR, task_name)
        demonstration_data_dir = "{}/tasks/{}/data/{}/".format(ASSETS_DIR, task_name, self.data_folder_name)

        starting_pose_path = "{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, task_name)
        self.demonstration_dir = demonstration_dir

    
        self.starting_pose = self.set_starting_pose() # set initial pose to give demonstration.
        self.demonstration, self.demonstration_states, self.gripper_states = self.robot.record_demonstration() 
        """
        TODO: You will need to implement your own demo recording function based on your robot hardware.

        The function should store wrist camera images, end-effector forces and end-effector poses in the demonstration directory.
        self.demonstration -> should include the sequence of end effector poses the one relative to the other. 
        i.e., action1 = np.linalg.inv(world_frame_pose1) @ world_frame_pose2
        self.demonstration_states -> should be a dictionary that includes the sequence of end effector forces and wrist camera images.

        self.demonstration = [pose1, pose2, pose3, ...] where pose1 is a 4x4 homogeneous transformation matrix.
        self.demonstration_states = {'data_imgs': [img1, img2, img3, ...], 'data_forces': [force1, force2, force3, ...], 'in_trajectory_identifier': [0,1,2,3...], 'is_on_waypoint': [1,1,1, ....]}
        The 'in_trajectory_identifier' is just the waypoint's number. So for the demo it is just a sequence of numbers from 0 to n. where n is the number of waypoints in the demonstration.
        It is used to fuse with the augmentation trajectories later on
        And  'is_on_waypoint': [1,1,1, ....] is always 1 for each demo waypoint it is used for the augmentation trajectories later on.
        
        self.gripper_states just 0 or 1 for open and close gripper states.

        Where to save:
        self.demonstration -> pkl.dump(self.demonstration, open("{}/tasks/{}/demonstration/recorded_demo.pkl".format(ASSETS_DIR, task_name), 'wb'))
        self.demonstration_states -> torch.save(self.demonstration_states, "{}/tasks/{}/data/demonstration_data.pt".format(ASSETS_DIR, task_name))
        self.starting_pose -> pkl.dump(self.starting_pose, open("{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, task_name), 'wb'))
        self.gripper_states -> pkl.dump(self.gripper_states, open("{}/tasks/{}/demonstration/gripper_states.pkl".format(ASSETS_DIR, task_name), 'wb'))
        
        Also:

        * you may find it useful to subsample your demonstration after collecting it, so that you have e.g., 50-60 demo waypoints.
        This would be equivalent to recording the demo waypoints at a low frequency.
        
        """

        """
        After recording the demonstration, you would need to reset the environment only once to the same
        initial state as before providing the demonstration.
         """ 
        self.robot.go_to_pose(self.starting_pose) # go to the inital demo pose to begin data collection.
        
        self.demonstration_in_base = copy.deepcopy(self.convert_demo_trajectory_to_base_frame())

        self.number_of_samples = number_of_samples
        self._data_imgs = torch.zeros(self.number_of_samples, 20, 3, 64, 64)
        self._data_forces = torch.zeros(self.number_of_samples, 20, 6)
        self._data_actions = torch.zeros(self.number_of_samples, 20, 4, 4)
        self._in_trajectory_identifier = torch.zeros(self.number_of_samples, 20, 1)
        self._number_data_collected = 0
        self._last_identifier = 0

    def save_data(self):
        if self._collect_last_inch:
            path = "{}/tasks/{}/data/last_inch/recorded_data.pt".format(ASSETS_DIR, self.task_name)
        else:
            path = "{}/tasks/{}/data/coarse/recorded_data_coarse.pt".format(ASSETS_DIR, self.task_name)

        data = {"data_imgs": self._data_imgs,
                "data_forces": self._data_forces,
                "data_actions": self._data_actions,
                "in_trajectory_identifier": self._in_trajectory_identifier,
                "number_data_collected": self._number_data_collected,
                "last_identifier": self._last_identifier}
        torch.save(data, path)

    def set_starting_pose(self):
        
        """
        TODO: Write a function where you move and set the initial pose from which you will
        record the demonstration."""
        pose = "TODO"
        pkl.dump(pose,
                 open("{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, self.task_name), 'wb'))

    def convert_demo_trajectory_to_base_frame(self):
        demo_in_base = []
        for i, pose in enumerate(self.demonstration):
            demo_in_base.append(self.starting_pose @ pose)
        return demo_in_base

    def get_random_transformation(self, linear_range=None, angular_range=None, coarse=False, last_inch=False):
        if linear_range is None:
            linear_range = [-.02, .02]
        if angular_range is None:
            angular_range = [-2, 2]
        angular_range[0] = np.deg2rad(angular_range[0])
        angular_range[1] = np.deg2rad(angular_range[1])
        linear = np.random.uniform(low=linear_range[0], high=linear_range[1], size=3)
        angular = np.random.uniform(low=angular_range[0], high=angular_range[1], size=3)
        transformation = np.eye(4)
        transformation[:3, 3] = linear
        transformation[:3, :3] = self.robot.se3_transforms.euler2rot("xyz", angular)
        return transformation

    def process_img(self, img):
        center_crop = CenterCrop(480)
        img_resize = Resize((128, 128))
        if img.shape[0] == img.shape[1]:
            img = img_resize(torch.tensor(img).permute(2, 0, 1) / 255)
        else:
            img = center_crop(torch.tensor(img).permute(2, 0, 1) / 255)
            img = img_resize(img)
        return img

    def compute_dino_similarity(self, demo_img):
        current_img = self.camera.get_rgb() 
        """
        TODO: You will have to implement your own get_rgb function based on your camera.
        """
        current_img = self.process_img(current_img).permute(1, 2, 0).cpu().numpy() * 255
        current_img = current_img.astype(np.uint8)
        demo_img = demo_img.astype(np.uint8)
        sim, image1_batch, image2_batch, extractor, similarities_patch_map, similarities = find_sim.sim(current_img,
                                                                                                        demo_img)
        image1 = (image1_batch[0].permute(1, 2, 0).cpu().numpy() * extractor.std) + extractor.mean
        image2 = (image2_batch[0].permute(1, 2, 0).cpu().numpy() * extractor.std) + extractor.mean

        if sim < SIMILARITY_THRESHOLD:
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(image1)
            ax[0, 0].set_title("Image 1")
            ax[0, 1].imshow(image2)
            ax[0, 1].set_title("Image 2")

            ax[1, 0].imshow(.5 * image1 + .5 * image2)
            ax[1, 0].set_title("Overlayed")
            sim_im = ax[1, 1].imshow(similarities_patch_map[:, :, 0].cpu().numpy(), cmap='gray')
            # Add colorbar only to last plot
            ax[1, 1].set_title("Cosine similarity: {:.2f}".format(torch.mean(similarities).item()))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(sim_im, cax=cbar_ax)
            # Remove grid around plots
            for i in range(2):
                for j in range(2):
                    ax[i, j].grid(False)
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
            plt.show()
        return sim

    """
    Collect augmentation trajectories in a self-supervised manner."""
    def collect_data(self):

        self.robot.go_to_pose(self.starting_pose) # go to the inital demo pose to begin data collection.
        data_collected = 0
        demo_length = len(self.demonstration_in_base)
        demo_index = 0
        trajectories_collected = 0
        trajectories_per_way_point = 10 # set as you like
        environment_disturbance_detected = False
        distance_errors = np.array([.0, .0, .0])
        rotation_errors = np.array([.0, .0, .0])

        while demo_index < demo_length and not environment_disturbance_detected:
            number_of_waypoint_data_collected = 0
            self._number_data_collected = data_collected
            self._last_identifier = demo_index
            if demo_index > 0: # if demo_index == 0 we are already at the first demo waypoint
                """
                Before moving to the next waypoint to collect data, first move to the previous waypoint to make sure
                that the next waypoint is always reachable!
                
                TODO: You will need to implement a replay_demonstration function based on your robot hardware.
                The replay demo should replay the demonstration up until the current waypoint index for which
                self-supervised data is collected."""

                self.robot.replay_demonstration(until_index=demo_index)

            """
            Sample the current waypoint demo."""
            current_demo_waypoint = self.demonstration_in_base[demo_index]

            """
            Copy the gripper action shown in the demonstration."""
            if self.gripper_states[demo_index] == 1:
                if self.robot.gripper_state == 0:
                    self.robot.grasp()
            elif self.gripper_states[demo_index] == 0:
                if self.robot.gripper_state == 1:
                    self.robot.open_gripper()


            pose_reached = copy.deepcopy(self.robot.get_eef_pose())
            demo_img = self.camera.get_rgb()  # Querying before processing to ensure data is in sync
            demo_img = self.process_img(demo_img).permute(1, 2, 0).cpu().numpy() * 255

            print(bcolors.OKBLUE + "[UPDATE] Moved to new waypoint: {}".format(demo_index))
            traj_num = 0
            while traj_num < trajectories_per_way_point:
                t_images = []
                t_forces = []
                t_identifiers = []
                t_actions = []
                t_is_on_waypoint = []
                gripper_states = []
                loop_time = time.time()
                if data_collected >= self.number_of_samples:
                    break

                """
                The following checks for the ENVIRONMENT DISTURBANCE validity condition."""
                similarity = self.compute_dino_similarity(demo_img)
                print(bcolors.OKBLUE + "[UPDATE] Similarity: {}".format(similarity) + bcolors.ENDC)
                if similarity < SIMILARITY_THRESHOLD and np.max(distance_errors) < .0010000001 and np.max(rotation_errors) < .7:
                    environment_disturbance_detected = True
                    print(bcolors.FAIL + "[STOPPING DATA COLLECTION] Environment disturbance detected." + bcolors.ENDC)
                    break

                """
                Sample a new random pose and go to it to collect data. """
                transformation = self.get_random_transformation()
                print(bcolors.OKBLUE + "[UPDATE] Transformation: {}".format(transformation[:3, 3]) + bcolors.ENDC)
                print("Transformation euler: {}".format(self.robot.se3_transforms.rot2euler("xyz", transformation[:3, :3], degrees=True)))
                new_pose = current_demo_waypoint @ transformation
                tracked_poses = False
                self.robot.go_to_pose_asynchronously(new_pose)

                """
                Record the trajectory followed so we can track it back to the current waypoint.
                If you feel a high force, break, and save the trajectory up to that point.

                TODO: You will need to implement a go_to_pose_asynchronously function based on your robot hardware.
                It needs to be asynchronous as while the robot is moving to the new pose, we will be collecting data.
                """

                traj_poses = []
                while not tracked_poses:
                    while self.robot.is_tracking_poses_async:
                        traj_poses.append(self.robot.get_eef_pose())
                        force_felt = self.robot.get_eef_wrench()
                        if np.abs(force_felt[0]) > 80 \
                                or np.abs(force_felt[1]) > 80 \
                                or np.abs(force_felt[2]) > 80 \
                                or np.abs(force_felt[3]) > 45 \
                                or np.abs(force_felt[4]) > 45 \
                                or np.abs(force_felt[5]) > 45:
                            print("Force threshold exceeded. Skipping sample", force_felt)
                            time.sleep(.1)
                    tracked_poses = True
                time.sleep(.1)
                """
                Go back to current_demo_waypoint we are collection data for
                to record the augmentation trajectory."""
                
                self.robot.go_to_pose_asynchronously(current_demo_waypoint)
                
                tracked_poses = False
                while not tracked_poses:
                    while self.robot.is_tracking_poses_async:
                        stime = time.time()

                        force_felt = self.robot.get_eef_wrench()
                        t_forces.append(copy.deepcopy(force_felt))
                        c_pose = self.robot.get_eef_pose()
                        t_actions.append(copy.deepcopy(c_pose))
                        temp_img = self.robot.camera.get_rgb()  # Querying before processing to ensure data is in sync
                        t_images.append(self.process_img(temp_img).cpu().numpy())
                        t_identifiers.append([demo_index])
                        gripper_states.append(self.gripper_states[demo_index])
                        if np.abs(pose_reached[0, 3] - c_pose[0, 3]) < .001 or np.abs( 
                                pose_reached[1, 3] - c_pose[1, 3]) < .001 or np.abs(
                            pose_reached[2, 3] - c_pose[2, 3]) < .001:
                            t_is_on_waypoint.append([1])
                        else:
                            t_is_on_waypoint.append([0])
                        if np.abs(force_felt[0]) > 80 \
                                or np.abs(force_felt[1]) > 80 \
                                or np.abs(force_felt[2]) > 80 \
                                or np.abs(force_felt[3]) > 45 \
                                or np.abs(force_felt[4]) > 45 \
                                or np.abs(force_felt[5]) > 45:
                            print("Force threshold exceeded. Skipping sample", force_felt)
                            time.sleep(.1)
                        data_collected += 1
                        number_of_waypoint_data_collected += 1
                        sleep_time = np.max((1 / self._recording_rate - (time.time() - stime), 0))
                        time.sleep(sleep_time)  # Sync loop to self._recording_rate Hz as recording
                    tracked_poses = True

                achieved_pose = self.robot.get_eef_pose()

                """
                THE FOLLOWING CHECKS FOR THE REACHABILITY CONDITION"""
                achieved_rot = self.robot.se3_transforms.rot2euler("xyz", achieved_pose[:3, :3], degrees=True)
                reached_rot = self.robot.se3_transforms.rot2euler("xyz", pose_reached[:3, :3], degrees=True)
                distance_errors = np.array([np.abs(achieved_pose[0, 3] - pose_reached[0, 3]), np.abs(
                    achieved_pose[1, 3] - pose_reached[1, 3]), np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3])])
                rotation_errors = np.array([np.abs(achieved_rot[0] - reached_rot[0]), np.abs(
                    achieved_rot[1] - reached_rot[1]), np.abs(
                    achieved_rot[2] - reached_rot[2])])
                if (np.abs(achieved_pose[0, 3] - pose_reached[0, 3]) > .001 or np.abs(
                        achieved_pose[1, 3] - pose_reached[1, 3]) > .001 or np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3]) > .001 or np.abs(achieved_rot[0] - reached_rot[0]) > .5
                        or np.abs(achieved_rot[1] - reached_rot[1]) > .5 or np.abs(achieved_rot[2] - reached_rot[2]) > .5):
                    print(
                        "Could not accurately return back to waypoint. Skipping sample. Returning back to starting pose and will move to waypoint again.")

                    """
                    If the Reachability condition is **NOT** met, discard augmentation trajectory, then go back to the starting pose
                    and replay the demonstration up until the current waypoint index for which we are collecting data."""
                    self.robot.go_to_pose(self.starting_pose)
                    self.robot.replay_demonstration(until_index=demo_index)
                    t_images, t_forces, t_identifiers, t_actions, t_is_on_waypoint, gripper_states = [], [], [], [], [], []
                    continue

                t_forces = np.array(t_forces)
                t_images = np.array(t_images)
                t_identifiers = np.array(t_identifiers)
                t_actions = np.array(t_actions)
                t_is_on_waypoint = np.array(t_is_on_waypoint)
                gripper_states = np.array(gripper_states)

                forces_torch = torch.from_numpy(t_forces).float()
                images_torch = torch.from_numpy(t_images).float()
                identifiers_torch = torch.from_numpy(t_identifiers).float()
                actions_torch = torch.from_numpy(t_actions).float()
                is_on_waypoint_torch = torch.from_numpy(t_is_on_waypoint).float()
                gripper_states_torch = torch.from_numpy(gripper_states).float()

                path = "{}/tasks/{}/data/{}/trajectory_sample_{}.pt".format(ASSETS_DIR, self.task_name, self.data_folder_name, trajectories_collected)

                data = {"data_imgs": images_torch, "data_forces": forces_torch, "data_actions": actions_torch, "in_trajectory_identifier": identifiers_torch, "gripper_state": gripper_states_torch, "is_on_waypoint": is_on_waypoint_torch}
                
                torch.save(data, path)
                trajectories_collected += 1
                traj_num += 1
                data_collection_details = {"number_of_trajectories_collected": trajectories_collected,
                                           "number_of_trajectories_per_way_point": trajectories_per_way_point,
                                           "demo_index": demo_index}
                path_2 = "{}/tasks/{}/data/{}/data_collection_details.pt".format(ASSETS_DIR, self.task_name, self.data_folder_name)
                torch.save(data_collection_details, path_2)
                print("Collected {} samples. Trajectories for this waypoint: {}. Demo index: {}. Time take: {}".format(
                    data_collected, traj_num, demo_index,
                    time.time() - loop_time))

            demo_index += 1  # After collecting all the trajectories for the current waypoint, move to the next.
        
        self.robot.go_to_pose(self.starting_pose)
    
        
    def collect_data_with_backtracking(self): # see pseudocode in supplementary

        self.robot.go_to_pose(self.starting_pose) # go to the inital demo pose to begin data collection.
        data_collected = 0
        demo_length = len(self.demonstration_in_base)
        demo_index = 0
        trajectories_collected = 0
        trajectories_per_way_point = 10 # set as you like
        environment_disturbance_detected = False
        distance_errors = np.array([.0, .0, .0])
        rotation_errors = np.array([.0, .0, .0])

        while demo_index < demo_length and not environment_disturbance_detected:
            number_of_waypoint_data_collected = 0
            self._number_data_collected = data_collected
            self._last_identifier = demo_index
            if demo_index > 0: # if demo_index == 0 we are already at the first demo waypoint
                """
                Before moving to the next waypoint to collect data, first move to the previous waypoint to make sure
                that the next waypoint is always reachable!
                
                TODO: You will need to implement a replay_demonstration function based on your robot hardware.
                The replay demo should replay the demonstration up until the current waypoint index for which
                self-supervised data is collected."""

                self.robot.replay_demonstration(until_index=demo_index)

            """
            Sample the current waypoint demo."""
            current_demo_waypoint = self.demonstration_in_base[demo_index]

            """
            Copy the gripper action shown in the demonstration."""
            if self.gripper_states[demo_index] == 1:
                if self.robot.gripper_state == 0:
                    self.robot.grasp()
            elif self.gripper_states[demo_index] == 0:
                if self.robot.gripper_state == 1:
                    self.robot.open_gripper()


            pose_reached = copy.deepcopy(self.robot.get_eef_pose())
            demo_img = self.camera.get_rgb()  # Querying before processing to ensure data is in sync
            demo_img = self.process_img(demo_img).permute(1, 2, 0).cpu().numpy() * 255

            print(bcolors.OKBLUE + "[UPDATE] Moved to new waypoint: {}".format(demo_index))
            traj_num = 0
            while traj_num < trajectories_per_way_point:
                t_images = []
                t_forces = []
                t_identifiers = []
                t_actions = []
                t_is_on_waypoint = []
                gripper_states = []
                loop_time = time.time()
                if data_collected >= self.number_of_samples:
                    break

                """
                The following checks for the ENVIRONMENT DISTURBANCE validity condition."""
                similarity = self.compute_dino_similarity(demo_img)
                print(bcolors.OKBLUE + "[UPDATE] Similarity: {}".format(similarity) + bcolors.ENDC)
                if similarity < SIMILARITY_THRESHOLD and np.max(distance_errors) < .0010000001 and np.max(rotation_errors) < .7:
                    environment_disturbance_detected = True
                    print(bcolors.FAIL + "[STOPPING DATA COLLECTION] Environment disturbance detected." + bcolors.ENDC)
                    break

                """
                Sample a new random pose and go to it to collect data. """
                transformation = self.get_random_transformation()
                print(bcolors.OKBLUE + "[UPDATE] Transformation: {}".format(transformation[:3, 3]) + bcolors.ENDC)
                print("Transformation euler: {}".format(self.robot.se3_transforms.rot2euler("xyz", transformation[:3, :3], degrees=True)))
                new_pose = current_demo_waypoint @ transformation
                tracked_poses = False
                self.robot.go_to_pose_asynchronously(new_pose)

                """
                Record the trajectory followed so we can track it back to the current waypoint.
                If you feel a high force, break, and save the trajectory up to that point.

                TODO: You will need to implement a go_to_pose_asynchronously function based on your robot hardware.
                It needs to be asynchronous as while the robot is moving to the new pose, we will be collecting data.
                """

                traj_poses = []
                tracked_trajectory = []
                while not tracked_poses:
                    while self.robot.is_tracking_poses_async:
                        traj_poses.append(self.robot.get_eef_pose())
                        tracked_trajectory.append(self.robot.get_controller_waypoint_target())
                        """
                        get_controller_waypoint_target() you need to implement a function based on your hardware
                        where you retrieve what your controller is tracking at each time step in order to have
                        the same behaviour when backtracking, e.g., to overcome large friction areas."""
                        force_felt = self.robot.get_eef_wrench()
                        if np.abs(force_felt[0]) > 80 \
                                or np.abs(force_felt[1]) > 80 \
                                or np.abs(force_felt[2]) > 80 \
                                or np.abs(force_felt[3]) > 45 \
                                or np.abs(force_felt[4]) > 45 \
                                or np.abs(force_felt[5]) > 45:
                            print("Force threshold exceeded. Skipping sample", force_felt)
                            time.sleep(.1)
                    tracked_poses = True
                time.sleep(.1)
                """
                Go back to current_demo_waypoint we are collection data by
                following backwards the trajectory followed when going
                to the random pose to record the augmentation trajectory. In free space
                this is identical to the collect_data() function, only when the robot
                complies with the environment and shapes the trajectory this makes a
                difference."""
                
                self.robot.go_to_pose_asynchronously(tracked_trajectory[-1])
                tracked_poses = False
                for backtrack_pose in tracked_trajectory[::-1]:
                    while self.robot.is_tracking_poses_async:
                        stime = time.time()

                        force_felt = self.robot.get_eef_wrench()
                        t_forces.append(copy.deepcopy(force_felt))
                        c_pose = self.robot.get_eef_pose()
                        t_actions.append(copy.deepcopy(c_pose))
                        temp_img = self.robot.camera.get_rgb()  # Querying before processing to ensure data is in sync
                        t_images.append(self.process_img(temp_img).cpu().numpy())
                        t_identifiers.append([demo_index])
                        gripper_states.append(self.gripper_states[demo_index])
                        if np.abs(pose_reached[0, 3] - c_pose[0, 3]) < .001 or np.abs( 
                                pose_reached[1, 3] - c_pose[1, 3]) < .001 or np.abs(
                            pose_reached[2, 3] - c_pose[2, 3]) < .001:
                            t_is_on_waypoint.append([1])
                        else:
                            t_is_on_waypoint.append([0])
                        if np.abs(force_felt[0]) > 80 \
                                or np.abs(force_felt[1]) > 80 \
                                or np.abs(force_felt[2]) > 80 \
                                or np.abs(force_felt[3]) > 45 \
                                or np.abs(force_felt[4]) > 45 \
                                or np.abs(force_felt[5]) > 45:
                            print("Force threshold exceeded. Skipping sample", force_felt)
                            time.sleep(.1)
                        data_collected += 1
                        number_of_waypoint_data_collected += 1
                        sleep_time = np.max((1 / self._recording_rate - (time.time() - stime), 0))
                        time.sleep(sleep_time)  # Sync loop to self._recording_rate Hz as recording
                        self.robot.go_to_pose_asynchronously(backtrack_pose)

                    tracked_poses = True

                achieved_pose = self.robot.get_eef_pose()

                """
                THE FOLLOWING CHECKS FOR THE REACHABILITY CONDITION"""
                achieved_rot = self.robot.se3_transforms.rot2euler("xyz", achieved_pose[:3, :3], degrees=True)
                reached_rot = self.robot.se3_transforms.rot2euler("xyz", pose_reached[:3, :3], degrees=True)
                distance_errors = np.array([np.abs(achieved_pose[0, 3] - pose_reached[0, 3]), np.abs(
                    achieved_pose[1, 3] - pose_reached[1, 3]), np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3])])
                rotation_errors = np.array([np.abs(achieved_rot[0] - reached_rot[0]), np.abs(
                    achieved_rot[1] - reached_rot[1]), np.abs(
                    achieved_rot[2] - reached_rot[2])])
                if (np.abs(achieved_pose[0, 3] - pose_reached[0, 3]) > .001 or np.abs(
                        achieved_pose[1, 3] - pose_reached[1, 3]) > .001 or np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3]) > .001 or np.abs(achieved_rot[0] - reached_rot[0]) > .5
                        or np.abs(achieved_rot[1] - reached_rot[1]) > .5 or np.abs(achieved_rot[2] - reached_rot[2]) > .5):
                    print(
                        "Could not accurately return back to waypoint. Skipping sample. Returning back to starting pose and will move to waypoint again.")

                    """
                    If the Reachability condition is **NOT** met, discard augmentation trajectory, then go back to the starting pose
                    and replay the demonstration up until the current waypoint index for which we are collecting data."""
                    self.robot.go_to_pose(self.starting_pose)
                    self.robot.replay_demonstration(until_index=demo_index)
                    t_images, t_forces, t_identifiers, t_actions, t_is_on_waypoint, gripper_states = [], [], [], [], [], []
                    continue

                t_forces = np.array(t_forces)
                t_images = np.array(t_images)
                t_identifiers = np.array(t_identifiers)
                t_actions = np.array(t_actions)
                t_is_on_waypoint = np.array(t_is_on_waypoint)
                gripper_states = np.array(gripper_states)

                forces_torch = torch.from_numpy(t_forces).float()
                images_torch = torch.from_numpy(t_images).float()
                identifiers_torch = torch.from_numpy(t_identifiers).float()
                actions_torch = torch.from_numpy(t_actions).float()
                is_on_waypoint_torch = torch.from_numpy(t_is_on_waypoint).float()
                gripper_states_torch = torch.from_numpy(gripper_states).float()

                path = "{}/tasks/{}/data/{}/trajectory_sample_{}.pt".format(ASSETS_DIR, self.task_name, self.data_folder_name, trajectories_collected)

                data = {"data_imgs": images_torch, "data_forces": forces_torch, "data_actions": actions_torch, "in_trajectory_identifier": identifiers_torch, "gripper_state": gripper_states_torch, "is_on_waypoint": is_on_waypoint_torch}
                
                torch.save(data, path)
                trajectories_collected += 1
                traj_num += 1
    
                print("Collected {} samples. Trajectories for this waypoint: {}. Demo index: {}. Time take: {}".format(
                    data_collected, traj_num, demo_index,
                    time.time() - loop_time))

            demo_index += 1  # After collecting all the trajectories for the current waypoint, move to the next.
        
        self.robot.go_to_pose(self.starting_pose)
    
   

if __name__ == '__main__':
    data_collector = DataCollector(task_name="test", data_collection_recording_rate=10)
    data_collector.collect_data() # this works best
    # or collect_data_with_backtracking() 
