import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset
import csv
import PIL.Image as Image
import os
import torchvision.transforms as T
from torch.autograd import Function
import pickle
import math


class SelectedCifar10(Dataset):
    def __init__(self, cifar10_dir, selected_images_csv, transform=None):
        super(SelectedCifar10, self).__init__()
        self.cifar10_dir = cifar10_dir
        self.data = []
        self.targets = []
        file_path = os.path.join(cifar10_dir, 'test_batch')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.transform = transform
        self.selected_images_csv = selected_images_csv
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        t_class, t_ind = map(int, self.selected_list[item])
        assert self.targets[t_ind] == t_class, 'Wrong targets in csv file.(line {})'.format(item+1)
        img, target = self.data[int(self.selected_list[item][1])], self.targets[t_ind]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.selected_list)

class Normalize(nn.Module):
    def __init__(self,):
        super(Normalize, self).__init__()
        self.ms = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - self.ms[0][i]) / self.ms[1][i]
        return x


def replace_relu_with_AS(model, AS_layer):
    for layer_idx in AS_layer:
        layer_idx_int = int(layer_idx)
        if isinstance(model[1].features.module[layer_idx_int], nn.ReLU):
            model[1].features.module[layer_idx_int] = AS()
 

class ASFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.relu(input)

    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        thou = input.min()
        x = input

        threshold = torch.tensor(1e-8, device=x.device, dtype=x.dtype)

        eps = torch.tensor(1e-10, device=x.device, dtype=x.dtype)
        beta = -torch.log((1 - threshold) / threshold + eps) / (thou + eps)

        sigmoid = 1 / (1 + torch.exp(-beta * x))

        grad_input = grad_output * sigmoid
        return grad_input

    

class AS(nn.Module):
    def __init__(self):
        super(AS, self).__init__()
    
    def forward(self, input):
        return ASFunction.apply(input)
