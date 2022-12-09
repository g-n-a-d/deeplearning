import torch
from torchvision.io import read_image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_raw):
        self.data_raw = data_raw/255
        self.frame_source = self.data_raw[:, 0, :, :, :]
        self.frame_driving = self.data_raw[:, 1, :, :, :]

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, index):
        return self.frame_source[index], self.frame_driving[index]

class Dataset_vc1(torch.utils.data.Dataset):
    def __init__(self, path_data_raw):
        with open(path_data_raw, 'r') as f:
            self.data_raw = f.readlines()

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, index):
        s = read_image(self.data_raw[index].split()[0])/255
        d = read_image(self.data_raw[index].split()[1])/255
        return s, d