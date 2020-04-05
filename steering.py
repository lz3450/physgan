#! /usr/bin/python

import os
import argparse
from time import strftime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from dataset import Udacity40G
from utils import save_model, load_model


class Dave2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # 3, 66, 200
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(inplace=True),

            # 24, 31, 98
            nn.Conv2d(in_channels=24, out_channels=36,
                      kernel_size=5, stride=2),
            nn.ELU(inplace=True),

            # 36, 14, 47
            nn.Conv2d(in_channels=36, out_channels=48,
                      kernel_size=5, stride=2),
            nn.ELU(inplace=True),

            # 48, 5, 22
            nn.Conv2d(in_channels=48, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ELU(inplace=True),

            # 64, 1, 18
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ELU(inplace=True)
        )

        self.linear = nn.Sequential(
            # 64, 1, 18
            nn.Linear(in_features=1152, out_features=100),
            nn.ELU(inplace=True),

            nn.Linear(in_features=100, out_features=50),
            nn.ELU(inplace=True),

            nn.Linear(in_features=50, out_features=10),
            nn.ELU(inplace=True),

            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


class Steering:
    path = 'steering_models'

    def __init__(self, device='cpu'):
        self.device = torch.device(device)

        self.model_left = Dave2()
        self.model_center = Dave2()
        self.model_right = Dave2()

        self._to_device()

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self._log("initialized.")

    def __call__(self, l, c, r):
        steering_center, steering_left, steering_right = self.model_center(l), self.model_left(c), self.model_right(r)
        return (steering_center + steering_left + steering_right) / 3

    def _log(self, msg):
        print(f"[{self.__class__.__qualname__}] {msg}")

    def _to_device(self):
        self.model_left = self.model_left.to(self.device)
        self.model_center = self.model_center.to(self.device)
        self.model_right = self.model_right.to(self.device)

    def load_model(self):
        import glob

        def get_newest_model(name):
            model_path = max(glob.glob(os.path.join(path, f'{name}-*')), key=os.path.getctime)
            self._log(f'model \"{model_path}\" loaded.')
            return model_path

        load_model(self.model_left, get_newest_model('left'))
        load_model(self.model_center, get_newest_model('center'))
        load_model(self.model_right, get_newest_model('right'))

        self._to_device()

    def train(self, epoch_number):

        self._log("start training ...")

        def _train(model, dataset, name):
            self._log(f"================ training {name} ================")

            dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

            criterion = nn.L1Loss()
            # criterion = nn.MSELoss(size_average=False)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            # optimizer = optim.Adam(net.parameters(), lr=0.001)

            model.train()
            for epoch in range(epoch_number):
                self._log(
                    f'---------------- epoch {epoch + 1} ----------------')

                loss_sum = 0.0
                for i, (scene, angle) in enumerate(dataloader):
                    scene, angle = scene.to(self.device), angle.to(self.device)
                    optimizer.zero_grad()
                    out = model(scene).view(-1)
                    loss = criterion(out, angle)
                    loss.backward()
                    optimizer.step()

                    self._log(f"training {name} {i + 1}: {loss.item():.4f}")

                    loss_sum += loss.item()

                # self._log(f"loss: {loss_sum / len(dataloader):.4f}")

            save_model(model.to(torch.device('cpu')), os.path.join(self.path, f"{name}-{strftime('%m-%d-%H%M%S')}.pkl"))

        _train(self.model_left, Udacity40G(t='left'), 'left')
        _train(self.model_center, Udacity40G(t='center'), 'center')
        _train(self.model_right, Udacity40G(t='right'), 'right')

    def test(self):

        def _test(model, dataset):
            dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=48, pin_memory=True)

            true_angle = []
            predicted_angle = []

            model.eval()
            for i, (scene, angle) in enumerate(dataloader):
                scene, angle = scene.to(self.device), angle.to(self.device)
                out = model(scene).view(-1)
                true_angle.append(angle.item())
                predicted_angle.append(out.item())
                self._log(f"{i + 1}: {out.item()}, {angle.item()}")

            return true_angle, predicted_angle

        self.load_model()

        with torch.no_grad():
            true_angle, predicted_angle = _test(self.model_left, Udacity40G(t='left'))
            true_angle, predicted_angle = _test(self.model_center, Udacity40G(t='center'))
            true_angle, predicted_angle = _test(self.model_right, Udacity40G(t='right'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(prog="Steering")

    arg_parser.add_argument('-d', '--device', type=str, help="Torch device.", default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])

    args = arg_parser.parse_args()

    print(args)

    steering_model = Steering(device=args.device)
    steering_model.train(50)
