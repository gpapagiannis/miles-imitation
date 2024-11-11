import copy
import torch
import numpy as np
from torch.utils.data import Dataset
import se3_tools as se3
import pickle as pkl
from utils import *
import matplotlib.pyplot as plt
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
import random
import time
from utils import *
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

ASSETS_DIR = "/path/to/assets"

class InteractionDataset(Dataset):
    def __init__(self,
                 path,
                 demonstration_path,
                 subsample_frequency=3,
                 horizon=10,
                 action_multiplier=3,
                 mix_trajectories=True,
                 normalize_actions=True,
                 mask_small_actions=False):

        assert action_multiplier == 3 or action_multiplier == 6
        self.normalize = False
        self.mix_trajectories = mix_trajectories
        self.action_multiplier = action_multiplier
        self.mask_small_actions = mask_small_actions
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"

        """Load the data and determine the length of the dataset
        based on the number of trajectories collected"""
        self.path = path
        self.demonstration_path = demonstration_path
        self.subsample_frequency = subsample_frequency
        self.horizon = horizon
        self.starting_pose = pkl.load(open("{}/starting_pose.pkl".format(self.demonstration_path), 'rb'))
        demo_path = "{}/recorded_demo.pkl".format(self.demonstration_path)  # Load demonstration data
        self.demonstration = pkl.load(open(demo_path, 'rb'))
        self.demonstration_in_base = torch.from_numpy(np.array(self.convert_demo_trajectory_to_base_frame())).to(
            self.device)
        print("Demonstration in base", self.demonstration_in_base.shape)

        collected_data_details = torch.load("{}/data_collection_details.pt".format(self.path))
        self.length =collected_data_details['number_of_trajectories_collected']
        self.last_demo_index_collected  = collected_data_details['demo_index']  + 1
        self.demonstration = self.demonstration[:self.last_demo_index_collected]
        self.demonstration_in_base  = self.demonstration_in_base[:self.last_demo_index_collected]
        print("Demonstration in base", self.demonstration_in_base[0]); input()
        self.demonstration_length = len(self.demonstration)  # Get the demonstration length to determine the longest sequence
        self.longest_sequence = self.get_longest_data_sequence()
        self.original_longer_sequence = copy.deepcopy(self.longest_sequence)
        if self.mix_trajectories:
            self.longest_sequence = np.array(
                [int(1.0 * self.longest_sequence[0])])  # TODO: 1.7 needs to be determined automatically
        print("[INTERACTION DATASET info] Longest sequence: {} | Demonstration length: {} | Number of samples: {}".format(self.longest_sequence[0], self.demonstration_length, self.length))
        self.demonstration_state_actions = torch.load("{}/demonstration_data.pt".format(self.path))  # Load the state - actions pairs observed in the demonstration


        self.demonstration_state_actions = {
            'data_imgs': self.demonstration_state_actions['data_imgs'][:self.last_demo_index_collected].to(self.device),
            'data_forces': self.demonstration_state_actions['data_forces'][:self.last_demo_index_collected].to(self.device),
            'data_actions': self.demonstration_in_base,
            'gripper_states': self.demonstration_state_actions['gripper_states'][:self.last_demo_index_collected].to(self.device),
            'in_trajectory_identifier': self.demonstration_state_actions['in_trajectory_identifier'][:self.last_demo_index_collected].to(self.device),
            'is_on_waypoint': self.demonstration_state_actions['is_on_waypoint'][:self.last_demo_index_collected].to(self.device)}

        # Determine the min and max normalization constants linear
        if normalize_actions:
            self.min_norm_consts_lin, self.max_norm_consts_lin = self.determine_min_max_normalization_constants_linear()

            if action_multiplier == 6:
                # Determine the min and max normalization constants angular
                self.min_norm_consts_ang, self.max_norm_consts_ang = self.determine_min_max_normalization_constants_orientation()

            self.normalize = normalize_actions # At the beginning of the class this is always set to False such that max, min normalization constants are determined without normalizing the actions

            # Save the normalization constants
            torch.save({'min_norm': self.min_norm_consts_lin, 'max_norm': self.max_norm_consts_lin}, "{}/normalization_constants_lin_{}.pt".format(self.path, horizon))

            if action_multiplier == 6:
                torch.save({'min_norm': self.min_norm_consts_ang, 'max_norm': self.max_norm_consts_ang}, "{}/normalization_constants_ang_{}.pt".format(self.path, horizon))

            if action_multiplier == 6:
                print("[INTERACTION DATASET info] [ORI]]]]]]]]]]]Min norm: {} | Max norm: {}".format(self.min_norm_consts_ang,
                                                                                                     self.max_norm_consts_ang))

            d_acts = self.process_actions(self.demonstration_state_actions['data_actions'])
            d_acts_masks = torch.ones_like(d_acts)
            d_acts_masks[torch.abs(d_acts) < 1e-3] = 0

            # Element by element multiplication of the actions with the masks
            d_acts = d_acts * d_acts_masks
            print("Demonstration actions", d_acts[self.subsample_frequency-1::self.subsample_frequency])
            print("max", torch.max(torch.abs(d_acts)))
        

    def convert_demo_trajectory_to_base_frame(self):
        demo_in_base = []
        for i, pose in enumerate(self.demonstration):
            demo_in_base.append(self.starting_pose @ pose)
        return demo_in_base

    def get_longest_data_sequence(self):
        """Get the longest data sequence in the dataset. Each trajectory contains the corrective trajectory
        to a corresponding waypoint determine in the in_trajectory_identifier. Hence, every trajectory's length
        is the length of the corrective trajectory + the length of the remaining demonstration trajectory
        including the current waypoint (i.e., self.demonstration_length - in_trajectory_identifier[0])."""
        longest_sequence = 0
        for i in range(self.length):
            # print("i", i)
            data = torch.load("{}/trajectory_sample_{}.pt".format(self.path, i))
            if np.floor((data['in_trajectory_identifier'].shape[0] + self.demonstration_length -
                         data['in_trajectory_identifier'][
                             0].cpu().numpy()) / self.subsample_frequency) > longest_sequence:
                longest_sequence = np.floor((data['in_trajectory_identifier'].shape[0] + self.demonstration_length -
                                             data['in_trajectory_identifier'][
                                                 0].cpu().numpy()) / self.subsample_frequency)
        return longest_sequence

    def show_trajectory(self, actions):
        """Show the trajectory of the end-effector in 3D space."""
        trajectory = torch.zeros((actions.shape[0], 3)).to(self.device)
        trajectory[0] = torch.tensor([0, 0, 0]).to(self.device)
        for i in range(actions.shape[0] - 1):
            trajectory[i + 1] = trajectory[i] + actions[i, :3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0].cpu().numpy(), trajectory[:, 1].cpu().numpy(), trajectory[:, 2].cpu().numpy())
        ax.plot(trajectory[:, 0].cpu().numpy(), trajectory[:, 1].cpu().numpy(), trajectory[:, 2].cpu().numpy(), 'ro')

        plt.show()

    def normalize_actions(self, data, normalize_ori=False):
        """Normalize the data in the range of [-1,+1] based on the min and max normalization constants.
        Data do not have batch dimension when normalized."""
        
        if not self.normalize:
            return data
        
        if normalize_ori:
            min = self.min_norm_consts_ang
            max = self.max_norm_consts_ang
            
        else:
            # Normalize every linear action separately
            min = self.min_norm_consts_lin
            max = self.max_norm_consts_lin
        data[:, 0::3] = 2 * (data[:, 0::3] - min[0]) / (max[0] - min[0]) - 1
        data[:, 1::3] = 2 * (data[:, 1::3] - min[1]) / (max[1] - min[1]) - 1
        data[:, 2::3] = 2 * (data[:, 2::3] - min[2]) / (max[2] - min[2]) - 1
        return data

    def denormalize_actions(self, data, denormalize_ori=False):
        """Denormalize the data in the range of [-1,+1] based on the min and max normalization constants.
        Note that this function denormalizes every action in data. So if we are denormalizing the linear part
        it will also denormalize the angular part. However, the angular part is discarded during training if
        we are training only for displacement and the linear part is discarded if we are training only for orientation."""
        if not self.normalize:
            return data
        if denormalize_ori:
            min = self.min_norm_consts_ang
            max = self.max_norm_consts_ang
      
        else:
            # Denormalize every linear action separately
            min = self.min_norm_consts_lin
            max = self.max_norm_consts_lin
        
        data[:, :, 0::3] = (data[:, :, 0::3] + 1) * (max[0] - min[0]) / 2 + min[0]
        data[:, :, 1::3] = (data[:, :, 1::3] + 1) * (max[1] - min[1]) / 2 + min[1]
        data[:, :, 2::3] = (data[:, :, 2::3] + 1) * (max[2] - min[2]) / 2 + min[2]

        return data

    def process_actions(self, actions):
        """Process actions such that they are relative to each other in the EEF frame.
        Creates a sequence of self.action_horizon to predict in the future."""
        actions_processed = torch.zeros((actions.shape[0], 6 * self.horizon)).to(self.device)
        for i in range(actions.shape[0] - 1):  # Leave the last action out, as it is always the identity matrix
            for k in range(self.horizon):
                if k + i + 1 < actions.shape[0]:
                    action_in_eef = np.linalg.inv(actions[i].cpu().numpy()) @ actions[i + k + 1].cpu().numpy()
                else:
                    action_in_eef = np.linalg.inv(actions[i].cpu().numpy()) @ actions[-1].cpu().numpy()
                action_linear = action_in_eef[:3, 3]
                action_angular = se3.rot2euler("xyz", action_in_eef[:3, :3])
                actions_processed[i, 3 * k: 3 * k + 3] = torch.from_numpy(action_linear).to(self.device)
                actions_processed[i, 3 * self.horizon + 3 * k: 3 * self.horizon + 3 * k + 3] = torch.from_numpy(
                        action_angular).to(self.device)
        # Set the last element to the identity matrix
        actions_processed[-1] = torch.zeros(6 * self.horizon).to(self.device)
        return actions_processed

    def process_gripper_states(self, gripper_states):
        """ Gripper states are processed such that they are repeated for the length of the horizon. This is done
        to be able to predict the gripper states in the future. """
        gripper_states_processed = torch.zeros((gripper_states.shape[0], self.horizon)).to(self.device)
        for i in range(gripper_states.shape[0]):
            for k in range(self.horizon):
                if k + i + 1 < gripper_states.shape[0]:
                    gripper_states_processed[i, k] = gripper_states[i + k + 1]
                else:
                    gripper_states_processed[i, k] = gripper_states[-1]
        return gripper_states_processed


    def dataset_distribution(self):
        """Plot the distribution of the dataset for every action direction individually"""
        x_actions, y_actions, z_actions = [], [], []
        for sample in range(self.length): # Go through every trajectory
            data = self.__getitem__(sample)
            actions = data['actions'][:, :3 * self.horizon] # Traj. length x (3 * self.horizon). Extract only linear actions
            actions_x = torch.zeros((actions.shape[0], actions.shape[1] // 3)).cpu() # Create a tensor to store x-dim actions only
            actions_y = torch.zeros((actions.shape[0], actions.shape[1] // 3)).cpu() # Create a tensor to store y-dim actions only
            actions_z = torch.zeros((actions.shape[0], actions.shape[1] // 3)).cpu() # Create a tensor to store z-dim actions only
            # Split the actions into x, y, z dimensions
            for i in range(actions.shape[0]):
                for j in range(actions.shape[1] // 3): # This is equal to self.horizon
                    actions_x[i, j] = actions[i, 3 * j]
                    actions_y[i, j] = actions[i, 3 * j + 1]
                    actions_z[i, j] = actions[i, 3 * j + 2]

            for a in actions_x:
                for a2 in a:
                    x_actions.append(a2)
            for a in actions_y:
                for a2 in a:
                    y_actions.append(a2)
            for a in actions_z:
                for a2 in a:
                    z_actions.append(a2)
        """Add labels to plots"""

        n, bins, patches = plt.hist(x_actions, bins=100)
        n = np.asanyarray(n)
        n = np.argmax(n)
        print("Bin with hihgest frequency: {}".format(bins[n]))
        plt.title("X actions")
        plt.show()
        plt.hist(y_actions, bins=100)
        plt.title("Y actions")
        plt.show()
        plt.hist(z_actions, bins=100)
        plt.title("Z actions")
        plt.show()

    def show_trajectory(self, actions, title, absolute=True, includes_horizon=False, initial_state=None, demonstration_actions=None):
        """Show the trajectory of the end-effector in 3D space.""" 
        # print("--------------------")
        absolute_poses = None
        if includes_horizon:
            absolute_poses = copy.deepcopy(initial_state)

            assert initial_state is not None
            actions = actions[::self.horizon]
            # print(actions); input()
            new_actions_lin = np.zeros((actions.shape[0] * self.horizon, 3))
            new_actions_ori = np.zeros((actions.shape[0] * self.horizon, 3))
            idx = 0
            for i in range(0, actions.shape[0]):
                for j in range(self.horizon):
                    new_actions_lin[idx] = actions[i, j * 3: j * 3 + 3].cpu().numpy()
                    new_actions_ori[idx] = actions[i, self.horizon * 3 + j * 3: self.horizon * 3 + j * 3 + 3].cpu().numpy()
                    idx+=1

            new_actions = np.zeros((new_actions_lin.shape[0], 4, 4))
            for i in range(new_actions_lin.shape[0]):
                new_actions[i] = np.eye(4)
                new_actions[i, :3, :3] = se3.euler2rot("xyz", new_actions_ori[i])
                new_actions[i, :3, 3] = new_actions_lin[i]


            new_actions_absolute = np.zeros((new_actions.shape[0]+1, 4, 4))
            initial_state = initial_state[0].cpu().numpy()
            new_actions_absolute[0] = initial_state
            b=0
            for a in range(new_actions.shape[0]):
                new_actions_absolute[a + 1] = initial_state @ new_actions[a]
                if (a + 1) % self.horizon == 0 and a != new_actions.shape[0] - 1:
                    b+= self.horizon
                    initial_state = new_actions_absolute[b]
                    # initial_state = absolute_poses[b-1].cpu().numpy()
            actions = torch.tensor(new_actions_absolute).float()

        trajectory = torch.zeros((actions.shape[0], 3))
        # if includes_horizon:
        if not absolute:
            init_pose = copy.deepcopy(actions[0])
            trajectory[0] = torch.tensor([0, 0, 0])
            for i in range(1, actions.shape[0]):
                trajectory[i] = (torch.inverse(init_pose) @ actions[i])[:3, 3]
        else:
            for i in range(actions.shape[0]):
                trajectory[i] = actions[i][:3, 3]

        if absolute_poses is not None:
            trajectory2 = torch.zeros((absolute_poses.shape[0], 3))
            for i in range(absolute_poses.shape[0]):
                trajectory2[i] = absolute_poses[i][:3, 3]

        if demonstration_actions is not None:
            trajectory3 = torch.zeros((demonstration_actions.shape[0], 3))
            for i in range(demonstration_actions.shape[0]):
                trajectory3[i] = demonstration_actions[i][:3, 3]
            # print("Traj difference: ", trajectory[:trajectory2.shape[0]] - trajectory2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0].cpu().numpy(), trajectory[:, 1].cpu().numpy(), trajectory[:, 2].cpu().numpy(), color='blue')
        ax.plot(trajectory[:, 0].cpu().numpy(), trajectory[:, 1].cpu().numpy(), trajectory[:, 2].cpu().numpy(), 'ro', color = 'blue')

        if absolute_poses is not None:
            ax.plot(trajectory2[:, 0].cpu().numpy(), trajectory2[:, 1].cpu().numpy(), trajectory2[:, 2].cpu().numpy(), 'ro', color='orange')
            ax.plot(trajectory2[:, 0].cpu().numpy(), trajectory2[:, 1].cpu().numpy(), trajectory2[:, 2].cpu().numpy(), color='orange')

        if demonstration_actions is not None:
            ax.plot(trajectory3[:, 0].cpu().numpy(), trajectory3[:, 1].cpu().numpy(), trajectory3[:, 2].cpu().numpy(), 'ro', color='red')
            ax.plot(trajectory3[:, 0].cpu().numpy(), trajectory3[:, 1].cpu().numpy(), trajectory3[:, 2].cpu().numpy(), color='red')
        ax.plot(trajectory[0:1, 0].cpu().numpy(), trajectory[0:1, 1].cpu().numpy(), trajectory[0:1, 2].cpu().numpy(), 'ro', color='green')

        # Set axis range to be equal to the max range
        max_range = np.array([trajectory[:, 0].cpu().numpy().max() - trajectory[:, 0].cpu().numpy().min(),
                              trajectory[:, 1].cpu().numpy().max() - trajectory[:, 1].cpu().numpy().min(),
                              trajectory[:, 2].cpu().numpy().max() - trajectory[:, 2].cpu().numpy().min()]).max() / 2.0
        
        mid_x = (trajectory[:, 0].cpu().numpy().max() + trajectory[:, 0].cpu().numpy().min()) * 0.5
        mid_y = (trajectory[:, 1].cpu().numpy().max() + trajectory[:, 1].cpu().numpy().min()) * 0.5
        mid_z = (trajectory[:, 2].cpu().numpy().max() + trajectory[:, 2].cpu().numpy().min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()
        

    def determine_min_max_normalization_constants_linear(self):

        min_x, max_x, min_y, max_y, min_z, max_z = 100, -100, 100, -100, 100, -100;
        
        for sample in range(self.length): # Go through every trajectory
            data = self.__getitem__(sample)
            
            actions = data['actions'][:, :3 * self.horizon] # Traj. length x (3 * self.horizon). Extract only linear actions

            actions_x = torch.zeros((actions.shape[0], actions.shape[1] // 3)).to(self.device) # Create a tensor to store x-dim actions only
            actions_y = torch.zeros((actions.shape[0], actions.shape[1] // 3)).to(self.device) # Create a tensor to store y-dim actions only
            actions_z = torch.zeros((actions.shape[0], actions.shape[1] // 3)).to(self.device) # Create a tensor to store z-dim actions only
            
            # Split the actions into x, y, z dimensions
            for i in range(actions.shape[0]):
                for j in range(actions.shape[1] // 3): # This is equal to self.horizon
                    actions_x[i, j] = actions[i, 3 * j]
                    actions_y[i, j] = actions[i, 3 * j + 1]
                    actions_z[i, j] = actions[i, 3 * j + 2]

            # Get x-dim min and max
            for action in actions_x:
                if torch.min(action) < min_x:
                    min_x = torch.min(action)
                if torch.max(action) > max_x:
                    max_x = torch.max(action)
            # Get y-dim min and max
            for action in actions_y:
                if torch.min(action) < min_y:
                    min_y = torch.min(action)
                if torch.max(action) > max_y:
                    max_y = torch.max(action)
            # Get z-dim min and max
            for action in actions_z:
                if torch.min(action) < min_z:
                    min_z = torch.min(action)
                if torch.max(action) > max_z:
                    max_z = torch.max(action)

        return torch.tensor([min_x, min_y, min_z]), torch.tensor([max_x, max_y, max_z])


    def determine_min_max_normalization_constants_orientation(self):
        min_x_ori, max_x_ori, min_y_ori, max_y_ori, min_z_ori, max_z_ori = 100, -100, 100, -100, 100, -100
        for sample in range(self.length): # Go through every trajectory
            data = self.__getitem__(sample)

            actions = data['actions'][:, 3 * self.horizon:2 * 3 * self.horizon] # Traj. length x (3 * self.horizon). Extract only angular actions. 2 * 3 * self.horizon is meant to exlcude the gripper state if it has been added

            actions_x = torch.zeros((actions.shape[0], actions.shape[1] // 3)).to(self.device) # Create a tensor to store x-dim actions only
            actions_y = torch.zeros((actions.shape[0], actions.shape[1] // 3)).to(self.device) # Create a tensor to store y-dim actions only
            actions_z = torch.zeros((actions.shape[0], actions.shape[1] // 3)).to(self.device) # Create a tensor to store z-dim actions only

            # Split the actions into x, y, z dimensions
            for i in range(actions.shape[0]):
                for j in range(actions.shape[1] // 3):
                    actions_x[i, j] = actions[i, 3 * j]
                    actions_y[i, j] = actions[i, 3 * j + 1]
                    actions_z[i, j] = actions[i, 3 * j + 2]

            # Get x-dim min and max
            for action in actions_x:
                if torch.min(action) < min_x_ori:
                    min_x_ori = torch.min(action)
                if torch.max(action) > max_x_ori:
                    max_x_ori = torch.max(action)

            # Get y-dim min and max
            for action in actions_y:
                if torch.min(action) < min_y_ori:
                    min_y_ori = torch.min(action)
                if torch.max(action) > max_y_ori:
                    max_y_ori = torch.max(action)

            # Get z-dim min and max
            for action in actions_z:
                if torch.min(action) < min_z_ori:
                    min_z_ori = torch.min(action)
                if torch.max(action) > max_z_ori:
                    max_z_ori = torch.max(action)

        return torch.tensor([min_x_ori, min_y_ori, min_z_ori]), torch.tensor([max_x_ori, max_y_ori, max_z_ori])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        stime = time.time()
        """Load the data for the idx trajectory and return the data"""
        data = torch.load("{}/trajectory_sample_{}.pt".format(self.path, idx))
        images = data['data_imgs'].to(self.device)
        forces = data['data_forces'].to(self.device)
        actions = data['data_actions'].to(self.device)
        gripper_states = data['gripper_state'].to(self.device)
        gripper_states = torch.unsqueeze(gripper_states, dim=-1)
        identifiers = data['in_trajectory_identifier'].to(self.device)
        is_on_way_point = 0 * data['is_on_waypoint'].to(self.device)
        is_on_way_point[-1] = 1

        """Append to the data the remaining demonstration from identifiers[0] to the end of the demonstration"""
        images = torch.cat([images, self.demonstration_state_actions['data_imgs'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        forces = torch.cat([forces, self.demonstration_state_actions['data_forces'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        actions = torch.cat([actions, self.demonstration_state_actions['data_actions'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        gripper_states = torch.cat([gripper_states, self.demonstration_state_actions['gripper_states'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        identifiers = torch.cat([identifiers, self.demonstration_state_actions['in_trajectory_identifier'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        is_on_way_point = torch.cat([is_on_way_point, self.demonstration_state_actions['is_on_waypoint'][int(identifiers[0][0].cpu().numpy()):]], dim=0)

        if images.shape[
            0] > self.subsample_frequency:  # Subsample only if the trajectory is longer than the subsample frequency
            images = images[self.subsample_frequency-1::self.subsample_frequency]
            forces = forces[self.subsample_frequency-1::self.subsample_frequency]
            actions = actions[self.subsample_frequency-1::self.subsample_frequency]
            gripper_states = gripper_states[self.subsample_frequency-1::self.subsample_frequency]
            identifiers = identifiers[self.subsample_frequency-1::self.subsample_frequency]
            is_on_way_point = is_on_way_point[self.subsample_frequency-1::self.subsample_frequency]
        # -----------------------#
        proprioception = torch.zeros((actions.shape[0], 7))
        for i in range(actions.shape[0]):
            rot = R.from_matrix(actions[i, :3, :3].cpu().numpy())
            rotvec = rot.as_rotvec()
            angle = np.linalg.norm(rotvec)
            axis = rotvec / angle
            ax_angle = np.concatenate((axis, np.array([angle])))
            proprioception[i] = torch.cat((actions[i, :3, 3].cpu(), torch.tensor(ax_angle)))
        proprioception = proprioception.to(self.device)
        # -----------------------#
        actions_unprocessed = copy.deepcopy(actions)
        actions = self.process_actions(copy.deepcopy(actions))

        """Repeat the last element of each variable until the length of the sequence is equal to self.longest_sequence
        using pytorch"""
        images = torch.cat([images, images[-1].repeat(int((self.longest_sequence - images.shape[0])[0]), 1, 1, 1)],dim=0)
        forces = torch.cat([forces, forces[-1].repeat(int((self.longest_sequence - forces.shape[0])[0]), 1)], dim=0)
        actions = torch.cat([actions, actions[-1].repeat(int((self.longest_sequence - actions.shape[0])[0]), 1)],dim=0)
        proprioception = torch.cat([proprioception, proprioception[-1].repeat(int((self.longest_sequence - proprioception.shape[0])[0]), 1)],dim=0)
        action_masks = torch.ones_like(actions)
        actions_unprocessed = torch.cat([actions_unprocessed, actions_unprocessed[-1].repeat(int((self.longest_sequence - actions_unprocessed.shape[0])[0]), 1, 1)],dim=0)

        # Zero out the actions with values less than 1e-3
        if self.mask_small_actions:
            if np.random.uniform(0, 1) < 0.5:
                action_masks[torch.abs(actions) < 1e-3] = 0
        gripper_states = torch.cat([gripper_states, gripper_states[-1].repeat(int((self.longest_sequence - gripper_states.shape[0])[0]), 1)],dim=0)
        # print("Grp states", gripper_states); input()
        identifiers = torch.cat([identifiers, identifiers[-1].repeat(int((self.longest_sequence - identifiers.shape[0])[0]), 1)],dim=0)
        is_on_way_point = torch.cat([is_on_way_point,is_on_way_point[-1].repeat(int((self.longest_sequence - is_on_way_point.shape[0])[0]), 1)],dim=0)


        augmentation_probability = np.random.uniform(0, 1)
        if augmentation_probability < 0.5:
            for img_batch in range(images.shape[0]):
                images[img_batch] = self.augment_image(images[img_batch])  # Augment images

        force_augmentation_probability = np.random.uniform(0, 1)
        if force_augmentation_probability < -0.2:
            for force_batch in range(forces.shape[0]):
                forces[force_batch] = self.augment_force(forces[force_batch])

        if self.normalize:
            actions[:, : 3 * self.horizon] = self.normalize_actions(actions[:, : 3 * self.horizon])
            if self.action_multiplier == 6:
                actions[:, 3 * self.horizon:] = self.normalize_actions(actions[:, 3 * self.horizon:], normalize_ori=True)

        # Concatenate the gripper_states at the end of the actions
        gripper_states  = self.process_gripper_states(gripper_states)
        if self.mix_trajectories:
                rand_num = np.random.uniform(0, 1)
                if rand_num < 1:  # With some probability shuffle a random trajectory from the dataset
                    rand_traj_idx = np.random.randint(0, self.length - 1)  # Shuffle random trajectory
                    rand_traj_data = self.get_trajectory(rand_traj_idx)
                    rand_idx_cut = np.random.randint(0, self.original_longer_sequence[
                        0] - 1)  # TODO: 15 needs to be set dynamically
                    rand_traj_images = rand_traj_data['images'][rand_idx_cut:]
                    rand_traj_forces = rand_traj_data['forces'][rand_idx_cut:]
                    rand_traj_actions = rand_traj_data['actions'][rand_idx_cut:]
                    rand_traj_unprocessed_actions = rand_traj_data['actions_unprocessed'][rand_idx_cut:]
                    rand_traj_identifiers = rand_traj_data['identifiers'][rand_idx_cut:]
                    rand_traj_is_on_way_point = rand_traj_data['is_on_way_point'][rand_idx_cut:]
                    rand_insertion_idx = np.random.randint(0, self.original_longer_sequence[0] - 1)
                    images[rand_idx_cut:] = rand_traj_images
                    forces[rand_idx_cut:] = rand_traj_forces
                    actions[rand_idx_cut:] = rand_traj_actions
                    identifiers[rand_idx_cut:] = rand_traj_identifiers
                    actions_unprocessed[rand_idx_cut:] = rand_traj_unprocessed_actions
                    is_on_way_point[rand_idx_cut:] = rand_traj_is_on_way_point


        return {'images': images, 
                'forces': forces, 
                'actions': actions, 
                'identifiers': identifiers, 
                'gripper_states': gripper_states,
                'proprioception': proprioception,
                'action_masks': action_masks,
                'actions_unprocessed': actions_unprocessed,
                'is_on_way_point': is_on_way_point}
    
    def get_trajectory(self, idx):
        stime = time.time()
        # print("Loading trajectory sample {}".format(idx))
        # print("Loading trajectory sample {}".format(idx))
        """Load the data for the idx trajectory and return the data"""
        data = torch.load("{}/trajectory_sample_{}.pt".format(self.path, idx))
        images = data['data_imgs'].to(self.device)
        forces = data['data_forces'].to(self.device)
        actions = data['data_actions'].to(self.device)
        identifiers = data['in_trajectory_identifier'].to(self.device)
        is_on_way_point = data['is_on_waypoint'].to(self.device)

        # Subsample the data to keep every 3rd element starting from the 3rd element

        """Append to the data the remaining demonstration from identifiers[0] to the end of the demonstration"""
        images = torch.cat(
            [images, self.demonstration_state_actions['data_imgs'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        forces = torch.cat(
            [forces, self.demonstration_state_actions['data_forces'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        actions = torch.cat(
            [actions, self.demonstration_state_actions['data_actions'][int(identifiers[0][0].cpu().numpy()):]], dim=0)
        identifiers = torch.cat([identifiers, self.demonstration_state_actions['in_trajectory_identifier'][
                                              int(identifiers[0][0].cpu().numpy()):]], dim=0)
        is_on_way_point = torch.cat([is_on_way_point, self.demonstration_state_actions['is_on_waypoint'][
                                                      int(identifiers[0][0].cpu().numpy()):]], dim=0)


        # print("[BEFORE] Actions shape: {}".format(actions.shape))
        if images.shape[
            0] > self.subsample_frequency:  # Subsample only if the trajectory is longer than the subsample frequency
            images = images[self.subsample_frequency::self.subsample_frequency]
            forces = forces[self.subsample_frequency::self.subsample_frequency]
            actions = actions[self.subsample_frequency::self.subsample_frequency]
            identifiers = identifiers[self.subsample_frequency::self.subsample_frequency]
            is_on_way_point = is_on_way_point[self.subsample_frequency::self.subsample_frequency]
        actions_unprocessed = copy.deepcopy(actions)
        actions = self.process_actions(copy.deepcopy(actions))
        # print("Actions shape: {}".format(actions.shape))
        """Repeat the last element of each variable until the length of the sequence is equal to self.longest_sequence
        using pytorch"""
        images = torch.cat(
            [images, images[-1].repeat(int((self.original_longer_sequence - images.shape[0])[0]), 1, 1, 1)],
            dim=0)
        forces = torch.cat([forces, forces[-1].repeat(int((self.original_longer_sequence - forces.shape[0])[0]), 1)],
                           dim=0)
        actions = torch.cat(
            [actions, actions[-1].repeat(int((self.original_longer_sequence - actions.shape[0])[0]), 1)],
            dim=0)
        identifiers = torch.cat(
            [identifiers, identifiers[-1].repeat(int((self.original_longer_sequence - identifiers.shape[0])[0]), 1)],
            dim=0)
        is_on_way_point = torch.cat(
            [is_on_way_point,
             is_on_way_point[-1].repeat(int((self.longest_sequence - is_on_way_point.shape[0])[0]), 1)],
            dim=0)
        actions_unprocessed = torch.cat(
            [actions_unprocessed, actions_unprocessed[-1].repeat(int((self.original_longer_sequence - actions_unprocessed.shape[0])[0]), 1, 1)],
            dim=0)
        return {'images': images, 'forces': forces, 'actions': actions, 'identifiers': identifiers,
                'is_on_way_point': is_on_way_point, 'actions_unprocessed': actions_unprocessed}
    def augment_image(self, im):
        ran_num = random.uniform(0, 1)
        im = np.copy(im.detach().cpu().numpy())
        # im = img_resize(torch.Tensor(im)).cpu().detach().numpy()
        if ran_num < .25:
            noise = .2
            b_rand = np.random.uniform(-noise, noise)
            g_rand = np.random.uniform(-noise, noise)
            r_rand = np.random.uniform(-noise, noise)
            im[0] += np.tile(b_rand, im.shape[1:])
            im[1] += np.tile(g_rand, im.shape[1:])
            im[2] += np.tile(r_rand, im.shape[1:])

        elif .25 <= ran_num < .50:
            im = adjust_contrast(torch.Tensor(im), random.uniform(.2, 1.5))  # (2)
            im_for_show = im.permute(1, 2, 0)
            im = im.cpu().detach().numpy()

        elif .50 <= ran_num < .75:
            im = adjust_brightness(torch.Tensor(im), random.uniform(0.1, 2))
            im_for_show = im.permute(1, 2, 0)
            im = im.cpu().detach().numpy()

        im = im.clip(0, 1)
        return torch.Tensor(im).to(self.device)

    def augment_force(self, force):
        ran_scaling = random.uniform(0.1, 1)
        return force * ran_scaling