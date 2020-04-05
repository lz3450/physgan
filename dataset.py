import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from numpy.linalg import solve
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from torchvision.transforms.functional import crop


class Udacity40G(Dataset):
    root = 'dataset/udacity-dataset/40G'
    image_ext = 'png'

    def __init__(self, t='center'):
        self.path = os.path.join(self.root, t)

        if not os.path.isdir(self.path):
            raise ValueError("udacity 40G dataset dose not have type '{}'.".format(t))

        self.steering_path = os.path.join(self.root, 'steering.csv')

        self.steering_data = pd.read_csv(self.steering_path, sep=',',
                                         dtype={'timestamp': np.int,
                                                'angle': np.float32,
                                                'torque': np.float32,
                                                'speed': np.float32})
        self.image_timestamps = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(self.path)])

        self.transform = Compose([
            Resize((150, 200)),
            Lambda(lambda scene: crop(scene, scene.size[1] - 66, 0, 66, 200)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.image_timestamps)

    def __getitem__(self, index):
        timestamp = self.image_timestamps[index]
        image_path = os.path.join(self.path, '{}.{}'.format(timestamp, self.image_ext))
        angle = self.steering_data.angle.loc[(self.steering_data.timestamp - timestamp).abs().idxmin()]

        with Image.open(image_path) as image:
            scene = self.transform(image)

        return scene, angle
