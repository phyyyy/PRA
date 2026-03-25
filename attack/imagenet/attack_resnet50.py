import os, sys
import yaml
import torch
import torchvision.transforms as T
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torchvision
import models as MODEL
from torch.backends import cudnn
import numpy as np
from utils import SelectedImagenet, Normalize, replace_relu_with_AS
import PIL
import timm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--method', type=str, default='PRA')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='adv/PRA')
parser.add_argument('--target_attack', default=False, action='store_true')
parser.add_argument(
    '--layer_config',
    type=str,
    default='./configs/imagenet_resnet50_layers.yaml',
    help='yaml file for md_pos / as_layer / decay_values'
)
args = parser.parse_args()


def load_layer_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    md_pos_list = cfg.get('md_pos', [])
    as_layer = cfg.get('as_layer', [])
    decay_values = cfg.get('decay_values', {})

    return md_pos_list, as_layer, decay_values


def hook_mdout(module, input, output):
    module.output = output


def get_hook_pd(ori_mdout, decay_rate):
    def hook_pd(module, input, output):
        modified_output = decay_rate * output + (1 - decay_rate) * ori_mdout
        module.new_output = output
        return modified_output
    return hook_pd


def _prep_hook(model, hooks, MD_pos_list, sigma, ori_img, iteration, device, decay_values):
    if sigma == 0 and iteration > 0:
        return
    for hook in hooks:
        hook.remove()
    hooks.clear()

    with torch.no_grad():
        for layer_pos in MD_pos_list:
            md_module = model[1]
            for part in layer_pos.split('.'):
                if part.isdigit():
                    md_module = md_module[int(part)]
                else:
                    md_module = getattr(md_module, part)

            mdout_hook = md_module.register_forward_hook(hook_mdout)
            model(ori_img + sigma * torch.randn(ori_img.size()).to(device))
            ori_mdout = md_module.output
            mdout_hook.remove()

            decay_rate = decay_values.get(layer_pos, 1.0)
            hook_func = get_hook_pd(ori_mdout, decay_rate)
            hook = md_module.register_forward_hook(hook_func)
            hooks.append(hook)


if __name__ == '__main__':
    args.epsilon = args.epsilon / 255

    MD_pos_list, AS_layer, decay_values = load_layer_config(args.layer_config)

    print(args)
    print('Loaded layer config from:', args.layer_config)
    print('MD_pos_list:', MD_pos_list)
    print('AS_layer:', AS_layer)
    print('decay_values:', decay_values)

    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    os.makedirs(args.save_dir, exist_ok=True)
    epsilon = args.epsilon
    batch_size = args.batch_size
    method = args.method
    save_dir = args.save_dir
    niters = args.niters
    target_attack = args.target_attack

    hooks = []
    sigma = 0.05

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trans = T.Compose([
        T.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
        T.ToTensor()
    ])

    dataset = SelectedImagenet(
        imagenet_val_dir='./ilsvrc2012/val/',
        selected_images_csv='./imagenet_labels/eval_seed201.csv',
        transform=trans
    )
    ori_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )

    model = MODEL.resnet.resnet50(state_dict_dir='./your_ckpt')

    model = nn.Sequential(
        Normalize(),
        model
    )
    model.eval()
    model.to(device)

    if target_attack:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()

    label_ls = []
    for ind, (ori_img, label) in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]

        ori_img = ori_img.to(device)
        img = ori_img.clone()
        m = 0

        for i in range(niters):
            if 'pgd' in method:
                img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
            else:
                img_x = img
            img_x.requires_grad_(True)

            if 'MD' in method or '_MD' in method:
                _prep_hook(model, hooks, MD_pos_list, sigma, ori_img, i, device, decay_values)
                model.zero_grad()

            if 'AS' in method or '_AS' in method:
                replace_relu_with_AS(model, AS_layer)

                att_out = model(img_x)
                pred = torch.argmax(att_out, dim=1).view(-1)
                loss = F.cross_entropy(att_out, label.to(device))
                input_grad = torch.autograd.grad(loss, img_x)[0].data

            if 'mdi2fgsm' in method or 'mifgsm' in method:
                input_grad = 1 * m + input_grad / (input_grad.abs().mean(dim=(1, 2, 3), keepdim=True))
                m = input_grad

            if target_attack:
                input_grad = -input_grad

            img = img.data + 1. / 255 * torch.sign(input_grad)
            img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
            img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
            img = torch.clamp(img, min=0, max=1)

        np.save(save_dir + '/adv_batch_{}.npy'.format(ind),
                torch.round(img.data * 255).cpu().numpy().astype(np.uint8()))
        np.save(save_dir + '/ori_batch_{}.npy'.format(ind),
                torch.round(ori_img.data * 255).cpu().numpy().astype(np.uint8()))
        del img, ori_img, input_grad
        print('batch_{}.npy saved'.format(ind))

    label_ls = torch.cat(label_ls)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('images saved')