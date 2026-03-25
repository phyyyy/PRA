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
from collections import defaultdict

# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        label_csv = open('./imagenet_label.csv', 'r')
        label_lists = list(csv.reader(label_csv))
        self.labels = {label_lists[i][0]:i for i in range(len(label_lists))}
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        imglist = list(reader)
        self.selected_list = [(self.labels[imglist[i][0]], imglist[i][0], "/".join([imglist[i][0]] + [imglist[i][j+1]])) for i in range(len(imglist)) for j in range(len(imglist[i])-1)]
    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)
    def __len__(self):
        return len(self.selected_list)

class Normalize(nn.Module):
    def __init__(self,):
        super(Normalize, self).__init__()
        self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - self.ms[0][i]) / self.ms[1][i]
        return x

def replace_relu_with_AS(model, AS_layer):
    for layer_path in AS_layer:
        AS_module = _get_AS_layer_module(model, layer_path)

        _replace_AS(AS_module)

def _replace_AS(module, parent_name=''):
    for child_name, child in module.named_children():
        full_name = parent_name + '.' + child_name if parent_name else child_name
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, AS())
        elif isinstance(child, nn.Sequential) or isinstance(child, nn.ModuleList):
            _replace_AS(child, full_name)


def _get_AS_layer_module(model, layer_path):
    base_module = model
    module = base_module[1]
    
    parts = layer_path.split('.')
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else: 
            module = getattr(module, part)

    return module


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

        threshold = torch.tensor(1e-9, device=x.device, dtype=x.dtype)

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
    