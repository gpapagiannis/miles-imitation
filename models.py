import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18

import torch.distributions as D


class LSTMPolicy(nn.Module):
    def __init__(self, action_dim=30,
                 lstm_layers=1,
                 lstm_hidden_units=1000,
                 vision_only=False,
                 force_only=False,
                 train_classifier=False,):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        if vision_only:
            self.vision_only = True
        else:
            self.vision_only = False
        
        if force_only:
            self.force_only = True
        else:
            self.force_only = False

        # self.device = "cpu"
        self.action_dim = action_dim
        self.train_classifier = train_classifier
        # self.unet1 = UNetBackbone(n_channels=3, n_classes=1, use_coordconv=False)
        # self.dsae_net = DSAENetwork(decouple_keypoints=False, split=None, is_5_dof=False,
                                            #    is_orientation=False, num_of_keypoints=32,
                                            #    im_size=128)
        self.resnet_backbone = resnet18(pretrained=False)
        linear_layer_in_fts = self.resnet_backbone.fc.in_features
        self.resnet_backbone.fc = nn.Linear(linear_layer_in_fts, 1000)

        self._rnn_type = 'LSTM'
        self.lstm_layers = lstm_layers
        self.lstm_hidden_units = lstm_hidden_units

        self.fc0 = nn.Linear(6, 100)
        self.prop_fc = nn.Linear(7, 100)
        if self.vision_only:
            self.lstm = nn.LSTM(input_size=1000,
                            hidden_size=self.lstm_hidden_units,
                            num_layers=lstm_layers,
                            batch_first=True)
        elif self.force_only:
            self.lstm = nn.LSTM(input_size=100,
                            hidden_size=self.lstm_hidden_units,
                            num_layers=lstm_layers,
                            batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=1100,
                            hidden_size=self.lstm_hidden_units,
                            num_layers=lstm_layers,
                            batch_first=True)

        action_heads = []
        if action_dim % 3 == 0:
            for _ in range(int(action_dim / 3)):
                action_heads.append(nn.Linear(1000, 3))
        else: # gripper actions are included
            for _ in range(int((action_dim - int(action_dim / 4)) / 3)):
                action_heads.append(nn.Linear(1000, 3))
            self.gripper_states_head = nn.Linear(1000, int(action_dim / 4))
        self.action_heads = nn.ModuleList(action_heads)
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        # self.fc3 = nn.Linear(100, action_dim)

        self.dropout = nn.Dropout(.25)

    def get_rnn_init_state(self, batch_size):
        if self._rnn_type == 'LSTM':
            return (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_units).to(self.device),
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_units).to(self.device))
        else:
            return torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_units).to(self.device)

    def forward(self, images, forces):
        # images_reshaped = images.view(-1, images.shape[2], images.shape[3], images.shape[4]) # B x T x C x H x W -> B * T x C x H x W
        # images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        # images = self.unet1(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        # images = self.dsae_net(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F

        # forces = F.relu(self.fc0(forces)) # B X T X 6 -> B x T x K

        # inp = torch.concat((images, forces), dim=2) # B x T x (F+K)


        if self.vision_only:
            images_reshaped = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
            inp = images
        elif self.force_only:
            forces = F.relu(self.fc0(forces))
            inp = forces
        else:
            images_reshaped = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
            forces = F.relu(self.fc0(forces))
            inp = torch.concat((images, forces), dim=2)


        hidden_state = self.get_rnn_init_state(inp.shape[0])
        x, hidden_state = self.lstm(inp, hidden_state) # B x T x H



        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x1 = F.relu(self.fc2(x))
        x1 = self.dropout(x1)
        # Pass x through evey action head
        action_heads_out = []
        for action_head in self.action_heads:
            action_heads_out.append(action_head(x1))
        action_heads_out = torch.cat(action_heads_out, dim=2)

        if self.action_dim % 3 != 0:
            gripper_states = self.gripper_states_head(x1)
            action_heads_out = torch.cat((action_heads_out, gripper_states), dim=2)
        # Concatenate the output of all action heads
        return action_heads_out


    def get_resnet_output(self, images, forces):
        # print(images.shape)
        images_reshaped = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
        images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1)
        return images
    def forward_step(self, images, forces, hidden_state):

        # images = self.unet1(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        # images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        # images = self.dsae_net(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        # print(images.shape)
        # input()

        if self.vision_only:
            images_reshaped = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
            inp = images
        elif self.force_only:
            forces = F.relu(self.fc0(forces))
            inp = forces
        else:
            images_reshaped = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
            forces = F.relu(self.fc0(forces))
            inp = torch.concat((images, forces), dim=2)

        if hidden_state is None:
            hidden_state = self.get_rnn_init_state(inp.shape[0])
        x, hidden_state = self.lstm(inp, hidden_state)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x1 = F.relu(self.fc2(x))
        x1 = self.dropout(x1)
        action_heads_out = []
        for action_head in self.action_heads:
            action_heads_out.append(action_head(x1))
        action_heads_out = torch.cat(action_heads_out, dim=2)

        if self.action_dim % 3 != 0:
            gripper_states = self.gripper_states_head(x1)
            action_heads_out = torch.cat((action_heads_out, gripper_states), dim=2)
        # Concatenate the output of all action heads
        return action_heads_out, hidden_state


    def get_lstm_output(self, images, forces, hidden_state_input=None, return_kpts=False):
        images_reshaped = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
        # images = self.unet1(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        # images = self.dsae_net(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F
        images = self.resnet_backbone(images_reshaped).view(images.shape[0], images.shape[1], -1) # B x T x F

        forces = F.relu(self.fc0(forces))
        inp = torch.concat((images, forces), dim=2)

        hidden_cell_states = torch.zeros((inp.shape[0], inp.shape[1], self.lstm_hidden_units)).to(
            self.device)  # B x T x H (to store the output of the LSTM's last layer cell state)
        state = 0  # 0: hidden state, 1: cell state
        hidden_state = copy.deepcopy(hidden_state_input)
        x = torch.zeros((inp.shape[0], inp.shape[1], self.lstm_hidden_units)).to(
            self.device)  # B x T x H (the input that will go into the MLP later)
        for batch_sample in range(inp.shape[0]):  # Sample through every sample in the batch
            # inp.shape: B x T x (F+6)
            if hidden_state_input is None:
                hidden_state = self.get_rnn_init_state(inp.shape[0])  # Init hidden state for every sample in the batch
            cell_state = hidden_state[state][
                self.lstm_layers - 1]  # Retrieve and store the last cell state of the current batch sample
            hidden_cell_states[batch_sample, 0] = cell_state[
                batch_sample]  # Store the last cell state of the current batch sample and first sample in the sequence of the current batch sample
            h0 = hidden_state[0][:, batch_sample:batch_sample + 1,
                 :].contiguous()  # Retrieve the hidden state of the current batch sample
            c0 = hidden_state[1][:, batch_sample:batch_sample + 1,
                 :].contiguous()  # Retrieve the cell state of the current batch sample
            hidden_state = (h0, c0)  # Pack into hidden state
            for sample in range(
                    inp.shape[1]):  # Sample through every sample in the sequence of the current batch sample
                x[batch_sample, sample], hidden_state = self.lstm(
                    inp[batch_sample:batch_sample + 1, sample:sample + 1],
                    hidden_state)  # Store the lstm output for every sample in the sequence of the current batch sample
                cell_state = hidden_state[state][self.lstm_layers - 1]
                hidden_cell_states[batch_sample, sample] = cell_state

            # hidden_state = self.get_rnn_init_state(inp.shape[0])
            # x, hidden_state = self.lstm(inp, hidden_state) # B x T x H
        if return_kpts:
            return x, images
        return hidden_cell_states

