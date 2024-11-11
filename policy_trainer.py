import copy
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset
import se3_tools as se3
import pickle as pkl
from models import LSTMPolicy
from utils import *
import matplotlib.pyplot as plt
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
import random
import time
from utils import *
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from miles_dataset_loader import InteractionDataset

ASSETS_DIR = "/path/to/assets"


class PolicyTrainer:
    def __init__(self,
                 task_name='test',
                 batch_size=8,
                 epochs=1000,
                 vision_only=False,
                 force_only=False,
                 action_horizon=10,
                 subsample_frequency=3,
                 predict_only_displacement=True,
                 with_gaussian_output=False,
                 train_classifier=False,
                 mix_trajectories=False,
                 normalize_actions=True,
                 train_ori=False,
                 mask_small_actions=False):
        self.train_classifier = train_classifier
        self.action_horizon = action_horizon
        self.epochs = epochs
        self.task_name = task_name
        self.predict_only_displacement = predict_only_displacement
        self.normalize_actions = normalize_actions
        self.train_ori = train_ori
        if predict_only_displacement:
            self.action_multiplier = 3
        else:
            self.action_multiplier = 6

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"

        self.data_folder = 'closed_loop'
        self.visions_only = vision_only
        self.force_only = force_only
        if vision_only:
            assert not force_only, "Cannot have vision only and force only"
        if force_only:
            assert not vision_only, "Cannot have vision only and force only"

        self.train_dataset = InteractionDataset(
            path="{}/tasks/{}/data/{}/".format(ASSETS_DIR, self.task_name, self.data_folder),
            demonstration_path="{}/tasks/{}/demonstration".format(ASSETS_DIR, self.task_name),
            horizon=action_horizon,
            action_multiplier=self.action_multiplier,
            subsample_frequency=subsample_frequency,
            mix_trajectories=mix_trajectories,
            normalize_actions=normalize_actions,
            mask_small_actions=mask_small_actions)

        self.train_loader, self.test_loader = torch.utils.data.random_split(self.train_dataset,
                                                                            [int(.8 * len(self.train_dataset)),
                                                                             self.train_dataset.length - int(
                                                                                 .8 * len(self.train_dataset))])
        self.train_loader = DataLoader(self.train_loader, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_loader, batch_size=batch_size, shuffle=True)

        make_dir("{}/tasks/{}/models".format(ASSETS_DIR, self.task_name))
        make_dir("{}/tasks/{}/models/closed_loop".format(ASSETS_DIR, self.task_name))
        make_dir("{}/tasks/{}/models/last_inch".format(ASSETS_DIR, self.task_name))

        self.model = LSTMPolicy(action_dim=3 * action_horizon, vision_only=vision_only, force_only=force_only,
                                with_gaussian_output=with_gaussian_output, train_classifier=train_classifier).to(
            self.device)
        if train_ori: # Adding this to include gripper states when predicting orientation. Again, lazy solution as everywhere else so remove it if gripper_states are to be predicted in a different way
            self.model = LSTMPolicy(action_dim=3 * action_horizon,
                                with_gaussian_output=with_gaussian_output, train_classifier=train_classifier).to(
            self.device)
        model_param_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model params : {model_param_num}')

    def pos_eval_metrics(self, pred, target):
        pos_error = torch.abs(torch.add(pred[:, :, :3 * self.action_horizon], -target[:, :, :3 * self.action_horizon]))

        x_error = None
        for i in range(self.action_horizon):
            if x_error is None:
                x_error = pos_error[:, :, 3 * i]
            else:
                x_error = torch.add(x_error, pos_error[:, :, 3 * i])

        x_error = torch.div(x_error, self.action_horizon)

        y_error = None
        for i in range(self.action_horizon):
            if y_error is None:
                y_error = pos_error[:, :, 3 * i + 1]
            else:
                y_error = torch.add(y_error, pos_error[:, :, 3 * i + 1])

        y_error = torch.div(y_error, self.action_horizon)

        z_error = None
        for i in range(self.action_horizon):
            if z_error is None:
                z_error = pos_error[:, :, 3 * i + 2]
            else:
                z_error = torch.add(z_error, pos_error[:, :, 3 * i + 2])

        z_error = torch.div(z_error, self.action_horizon)
        "Extract all non zero values from the tensors"
        # x_error = x_error[x_error != 0]
        # y_error = y_error[y_error != 0]
        # z_error = z_error[z_error != 0]
        return torch.mean(x_error), torch.mean(y_error), torch.mean(z_error)
        # return torch.mean(pos_error[:, :, 0]), torch.mean(pos_error[:, :, 1]), torch.mean(pos_error[:, :, 2])

    def ori_eval_metrics(self, pred, target):
        ori_error = torch.abs(torch.add(pred[:,:,:3 * self.action_horizon], -target[:, :, 3 * self.action_horizon:2 * 3 * self.action_horizon])) # 3 * self.action_horizon is meant to exlcude the gripper state if it has been added
        x_ori_error = None
        for i in range(self.action_horizon):
            if x_ori_error is None:
                x_ori_error = ori_error[:, :, 3 * i]
            else:
                x_ori_error = torch.add(x_ori_error, ori_error[:, :, 3 * i])

        x_ori_error = torch.div(x_ori_error, self.action_horizon)

        y_ori_error = None
        for i in range(self.action_horizon):
            if y_ori_error is None:
                y_ori_error = ori_error[:, :, 3 * i + 1]
            else:
                y_ori_error = torch.add(y_ori_error, ori_error[:, :, 3 * i + 1])

        y_ori_error = torch.div(y_ori_error, self.action_horizon)

        z_ori_error = None
        for i in range(self.action_horizon):
            if z_ori_error is None:
                z_ori_error = ori_error[:, :, 3 * i + 2]
            else:
                z_ori_error = torch.add(z_ori_error, ori_error[:, :, 3 * i + 2])

        z_ori_error = torch.div(z_ori_error, self.action_horizon)

        return torch.mean(x_ori_error) * 180 / np.pi,\
            torch.mean(y_ori_error) * 180 / np.pi,\
            torch.mean(z_ori_error) * 180 / np.pi
        # return 180 * torch.mean(pos_error[:, :, 0]) * 180 / np.pi, 180 * torch.mean(
        #     pos_error[:, :, 1]) / np.pi, 180 * torch.mean(pos_error[:, :, 2]) / np.pi

    def save_model(self):
        extension=""
        if self.visions_only:
            extension = "_vision_only"
        elif self.force_only:
            extension = "_force_only"

        if self.train_ori:
            torch.save(self.model.state_dict(),
                       "{}/tasks/{}/models/{}/policy_lstm_seq_{}_ori{}.pt".format(ASSETS_DIR, self.task_name,
                                                                               self.data_folder, self.action_horizon, extension))
        else:
            torch.save(self.model.state_dict(),
                   "{}/tasks/{}/models/{}/policy_lstm_seq_{}_lin{}.pt".format(ASSETS_DIR, self.task_name,
                                                                                  self.data_folder, self.action_horizon, extension))
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='mean')

        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            running_loss = 0
            ev_x, ev_y, ev_z, ev_ex, ev_ey, ev_ez, ev_gp = 0, 0, 0, 0, 0, 0, 0
            gp_error = 0
            self.model.train()
            for i, data in enumerate(self.train_loader):
                # print("Masks shape: ", data['action_masks'].shape)
                optimizer.zero_grad()
                output = self.model(data['images'], data['forces'])
                if self.train_ori:
                    loss = criterion(output, data['actions'][:, :, 3 * self.action_horizon:])
                else:
                    loss = criterion(output, 
                                     data['actions'][:, :, :3 * self.action_horizon])
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                if not self.train_ori:

                    er_x, er_y, er_z = self.pos_eval_metrics(
                        self.train_dataset.denormalize_actions(output), 
                        self.train_dataset.denormalize_actions(data['actions']))
                    ev_x += er_x
                    ev_y += er_y
                    ev_z += er_z

                if not self.predict_only_displacement:
                    # print("ACTS", data['actions'])
                    # print("--------1--------")
                    er_ex, er_ey, er_ez = self.ori_eval_metrics(self.train_dataset.denormalize_actions(output, denormalize_ori=True), 
                                                                self.train_dataset.denormalize_actions(data['actions'], denormalize_ori=True))
                    # print("--------END--------")
                    ev_ex += er_ex
                    ev_ey += er_ey
                    ev_ez += er_ez
          
                # if i % 10 == 9:
                print(bcolors.OKBLUE +
                          '[%d, %5d] Loss: %.3f | Error x: %.3f | Error y: %.3f| Error z: %.3f | AnError x: %.3f | AnError y: %.3f| AnError z: %.3f | Error GP: %.3f ' %
                          (epoch + 1, i + 1, running_loss / (i + 1), ev_x / (i + 1), ev_y / (i + 1), ev_z / (i + 1),
                           ev_ex / (i + 1),
                           ev_ey / (i + 1), ev_ez / (i + 1), ev_gp / (i + 1)) + bcolors.ENDC)
            if epoch % 10 == 0 and epoch > 0:
                for p in optimizer.param_groups:
                    p['lr'] = .95 * p['lr']
                    print("Reducing network lr: ", p['lr'])

            if epoch % 10 == 9:
                self.save_model()
                self.evaluate()


    def evaluate(self):
        with torch.no_grad():
            self.model.eval()
            running_loss = 0
            criterion = torch.nn.MSELoss(reduction='mean')

            iters, ev_x, ev_y, ev_z, ev_ex, ev_ey, ev_ez, ev_gp = 0, 0, 0, 0, 0, 0, 0, 0

            for i, data in enumerate(self.test_loader):

                output = self.model(data['images'], data['forces'])  # for m in range(output.shape[0]):
                # print(output)
    
                if self.train_ori:
                    loss = criterion(output, data['actions'][:,:, 3 * self.action_horizon:])
                else:
                    loss = criterion(output, data['actions'][:,:, :3 * self.action_horizon])

                running_loss += loss.item()
 
                if not self.train_ori:
                    er_x, er_y, er_z = self.pos_eval_metrics(self.train_dataset.denormalize_actions(output),
                                                         self.train_dataset.denormalize_actions(data['actions']))
                    ev_x += er_x
                    ev_y += er_y
                    ev_z += er_z
                if not self.predict_only_displacement:
                    er_ex, er_ey, er_ez = self.ori_eval_metrics(self.train_dataset.denormalize_actions(output, denormalize_ori=True),
                                                         self.train_dataset.denormalize_actions(data['actions'], denormalize_ori=True))
                    ev_ex += er_ex
                    ev_ey += er_ey
                    ev_ez += er_ez



                iters += 1
            print(bcolors.WARNING + bcolors.BOLD +
                  "Evaluation Loss: %.3f | Error x: %.3f | Error y: %.3f | Error z: %.3f | AnError x: %.3f | AnError y: %.3f| AnError z: %.3f | Error GP: %.3f" % (
                      running_loss / iters, ev_x / iters, ev_y / iters, ev_z / iters, ev_ex / iters, ev_ey / iters, 
                      ev_ez / iters, ev_gp / iters) + bcolors.ENDC)
    
    def evaluate_trajectories_visually(self):
        self.model_lin = LSTMPolicy(action_dim=3 * self.action_horizon).to(self.device)
        self.model_ori = LSTMPolicy(action_dim=3 * self.action_horizon,).to(self.device)
        self.model_lin.load_state_dict(torch.load("{}/tasks/{}/models/{}/policy_lstm_seq_{}_lin.pt".format(ASSETS_DIR, self.task_name,
                                                                                       self.data_folder, self.action_horizon)))
        self.model_ori.load_state_dict(torch.load("{}/tasks/{}/models/{}/policy_lstm_seq_{}_ori.pt".format(ASSETS_DIR, self.task_name,
                                                                                       self.data_folder, self.action_horizon)))
        self.model_lin.eval()
        self.model_ori.eval()
        with torch.no_grad():
            self.model.eval()
            running_loss = 0
            criterion = torch.nn.MSELoss(reduction='mean')

            iters, ev_x, ev_y, ev_z, ev_ex, ev_ey, ev_ez, ev_gp = 0, 0, 0, 0, 0, 0, 0, 0

            for i, data in enumerate(self.train_loader):

                output = self.model_lin(data['images'], data['forces'])  # for m in range(output.shape[0]):
                output_ori = self.model_ori(data['images'], data['forces'])  # for m in range(output.shape[0]):
                output_copy = torch.clone(output)
                output_ori_copy = torch.clone(output_ori)
                o_with_ori = torch.cat([self.train_dataset.denormalize_actions(output_copy.detach()),
                                             self.train_dataset.denormalize_actions(output_ori_copy, denormalize_ori=True)], dim=-1)
                for o in range(o_with_ori.shape[0]):
                    self.train_dataset.show_trajectory(actions=o_with_ori[o], 
                                                       title="Trajectory_sample_{}_ACTIONS_PROCESSED".format(i * o), 
                                                       includes_horizon=True, 
                                                       initial_state=data['actions_unprocessed'][o],)
                                                    #    demonstration_actions=self.train_dataset.demonstration_state_actions['data_actions'])

    
                if self.train_ori:
                    loss = criterion(output, data['actions'][:,:, 3 * self.action_horizon:])
                else:
                    
                    loss = criterion(output, data['actions'][:,:, :3 * self.action_horizon])

                running_loss += loss.item()


                if not self.train_ori:
                    er_x, er_y, er_z = self.pos_eval_metrics(self.train_dataset.denormalize_actions(output),
                                                         self.train_dataset.denormalize_actions(data['actions']))
                    ev_x += er_x
                    ev_y += er_y
                    ev_z += er_z
                if not self.predict_only_displacement:
                    er_ex, er_ey, er_ez = self.ori_eval_metrics(self.train_dataset.denormalize_actions(output, denormalize_ori=True),
                                                         self.train_dataset.denormalize_actions(data['actions'], denormalize_ori=True))
                    ev_ex += er_ex
                    ev_ey += er_ey
                    ev_ez += er_ez


                iters += 1
            print(bcolors.WARNING + bcolors.BOLD +
                  "Evaluation Loss: %.3f | Error x: %.3f | Error y: %.3f | Error z: %.3f | AnError x: %.3f | AnError y: %.3f| AnError z: %.3f | Error GP: %.3f" % (
                      running_loss / iters, ev_x / iters, ev_y / iters, ev_z / iters, ev_ex / iters, ev_ey / iters, 
                      ev_ez / iters, ev_gp / iters) + bcolors.ENDC)


if __name__ == '__main__':
    trainer = PolicyTrainer(task_name="test",
                            action_horizon=5,
                            predict_only_displacement=False,
                            vision_only=False,
                            force_only=True,
                            subsample_frequency=1,
                            with_gaussian_output=False,
                            train_classifier=False,
                            mix_trajectories=False,
                            train_ori=True,
                            normalize_actions=True,
                            mask_small_actions=False)

    trainer.train()
    

