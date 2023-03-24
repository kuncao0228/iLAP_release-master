import os
from skimage import io, transform
from PIL import Image
import torch

import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean


class IncrementalDataSet(Dataset):

    def __init__(self, label, rootDir, transform=None):
        self.label = label
        self.root_dir = rootDir
        self.transform = transform
        self.samples = []
        for fileName in os.listdir(self.root_dir):
            # if torch.is_tensor(idx):
            #     idx = idx.tolist()
            # fileName = os.listdir(self.root_dir)[idx]
            img_name = os.path.join(self.root_dir, fileName)
            image = Image.open(img_name).convert("RGB")
            image= image.resize((32, 32))
            if self.transform:
                image = self.transform(image)

#             print(image.shape)
#             image = np.transpose(image, (2,0,1))

            # TO DEVICE?
#             image = torch.from_numpy(image).float()

            sample = (image, self.label)
            
            self.samples.append(sample)


    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):

        return self.samples[idx]
