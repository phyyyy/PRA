import os, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import models
import torchvision.datasets as DATASETS
import argparse
import numpy as np
import logging

import models.vgg_pytorch


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--log-dir', type=str, default='results.log')
args = parser.parse_args()
print(args)
logging.basicConfig(filename=args.log_dir, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

cudnn.benchmark = False
cudnn.deterministic = True
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

target = torch.from_numpy(np.load(args.dir + '/labels.npy')).long()
if 'target' in args.dir:
    label_switch = torch.tensor([1,2,3,4,5,6,7,8,9,0]).long()
    target = label_switch[target]


gdas = models.__dict__['gdas']('./ckpt/gdas-cifar10-best.pth')
gdas.to(device)
gdas.eval()

pyramidnet = models.__dict__['pyramidnet272'](num_classes = 10)
pyramidnet.load_state_dict(torch.load('./ckpt/pyramidnet272-checkpoint.pth', map_location=device)['state_dict'])
pyramidnet.to(device)
pyramidnet.eval()

ResNeXt_29_8_64d = models.__dict__['resnext'](
                cardinality=8,
                num_classes=10,
                depth=29,
                widen_factor=4,
                dropRate=0,
            )
ResNeXt_29_8_64d = nn.DataParallel(ResNeXt_29_8_64d)
ResNeXt_29_8_64d.load_state_dict(torch.load('./ckpt/resnext-8x64d/model_best.pth.tar', map_location=device)['state_dict'])
ResNeXt_29_8_64d.eval()

DenseNet_BC_L190_k40 = models.__dict__['densenet'](
                num_classes=10,
                depth=190,
                growthRate=40,
                compressionRate=2,
                dropRate=0,
            )
DenseNet_BC_L190_k40 = nn.DataParallel(DenseNet_BC_L190_k40)
DenseNet_BC_L190_k40.load_state_dict(torch.load('./ckpt/densenet-bc-L190-k40/model_best.pth.tar', map_location=device)['state_dict'])
DenseNet_BC_L190_k40.eval()

WRN = models.__dict__['wrn'](
                num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,
            )
WRN = nn.DataParallel(WRN)
WRN.load_state_dict(torch.load('./ckpt/WRN-28-10-drop/model_best.pth.tar', map_location=device)['state_dict'])
WRN.eval()

vgg = models.vgg_pytorch.vgg19_bn(pretrained=False)
checkpoint_path_vgg = './ckpt/vgg19_bn/vgg19_bn.pt'
vgg.load_state_dict(torch.load(checkpoint_path_vgg, map_location=device))
vgg.to(device)
vgg.eval()

incv3 = models.inception_v3(pretrained=False)
checkpoint_path_incv3 = './ckpt/inception_v3.pt'
incv3.load_state_dict(torch.load(checkpoint_path_incv3, map_location=device))
incv3.to(device)
incv3.eval()

mobilv2 = models.mobilenet_v2(pretrained=False)
chepoint_path_mobilv2 = './ckpt/mobilenet_v2.pt'
mobilv2.load_state_dict(torch.load(chepoint_path_mobilv2, map_location=device))
mobilv2.to(device)
mobilv2.eval()

resnet18 = models.resnet18(pretrained=False)
checkpoint_path_resnet18 = './ckpt/resnet18.pt'
resnet18.load_state_dict(torch.load(checkpoint_path_resnet18, map_location=device))
resnet18.to(device)
resnet18.eval()

def get_pred(model, img, advs):
    return torch.argmax(model(img), dim=1).view(1,-1), torch.argmax(model(advs), dim=1).view(1,-1)
def normal(x):
    ms = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    for i in range(x.shape[1]):
        x[:,i,:,:] = (x[:,i,:,:] - ms[0][i]) / ms[1][i]
    return x

vgg_correct = 0
WRN_correct = 0
ResNeXt_29_8_64d_correct = 0
DenseNet_BC_L190_k40_correct = 0
pyramidnet_correct = 0
gdas_correct = 0
incv3_correct = 0
mobilv2_correct = 0
resnet18_correct = 0

vgg_success = 0
WRN_success = 0
ResNeXt_29_8_64d_success = 0
DenseNet_BC_L190_k40_success = 0
pyramidnet_success = 0
gdas_success = 0
incv3_success = 0
mobilv2_success = 0
resnet18_success = 0

advfile_ls = os.listdir(args.dir)
img_num = 0

ori_files = [f for f in os.listdir(args.dir) if 'ori_batch' in f]
num_batches = len(ori_files)
for batch_index in range(num_batches):
    adv_batch_path = os.path.join(args.dir, f'adv_batch_{batch_index}.npy')
    adv_batch = torch.from_numpy(np.load(adv_batch_path)).float() / 255

    ori_batch_path = os.path.join(args.dir, f'ori_batch_{batch_index}.npy')
    ori_batch = torch.from_numpy(np.load(ori_batch_path)).float() / 255
    if batch_index == 0:
        batch_size = adv_batch.shape[0]
    img_num += adv_batch.shape[0]

    labels = target[batch_index * batch_size : (batch_index * batch_size + adv_batch.shape[0])]
    adv_inputs = normal(adv_batch.clone())
    ori_inputs = normal(ori_batch.clone())
    adv_inputs, ori_inputs, labels = adv_inputs.to(device), ori_inputs.to(device), labels.to(device)

    with torch.no_grad():
        WRN_accuracy, WRN_successrate = get_pred(WRN, ori_inputs, adv_inputs)
        ResNeXt_29_8_64d_accuracy, ResNeXt_29_8_64d_successrate = get_pred(ResNeXt_29_8_64d, ori_inputs, adv_inputs)
        DenseNet_BC_L190_k40_accuracy, DenseNet_BC_L190_k40_successrate = get_pred(DenseNet_BC_L190_k40, ori_inputs, adv_inputs)
        pyramidnet_accuracy, pyramidnet_successrate = get_pred(pyramidnet, ori_inputs, adv_inputs)
        gdas_accuracy, gdas_successrate = get_pred(gdas, ori_inputs, adv_inputs)
        vgg_accuracy, vgg_successrate = get_pred(vgg, ori_inputs, adv_inputs)
        incv3_accuracy, incv3_successrate = get_pred(incv3, ori_inputs, adv_inputs)
        mobilv2_accuracy, mobilv2_successrate = get_pred(mobilv2, ori_inputs, adv_inputs)
        resnet18_accuracy, resnet18_successrate = get_pred(resnet18, ori_inputs, adv_inputs)

    WRN_correct += (labels == WRN_accuracy.squeeze(0)).sum().item()
    WRN_success += ((labels == WRN_accuracy.squeeze(0))*(labels != WRN_successrate.squeeze(0))).sum().item()
    ResNeXt_29_8_64d_correct += (labels == ResNeXt_29_8_64d_accuracy.squeeze(0)).sum().item()
    ResNeXt_29_8_64d_success += ((labels == ResNeXt_29_8_64d_accuracy.squeeze(0))*(labels != ResNeXt_29_8_64d_successrate.squeeze(0))).sum().item()
    DenseNet_BC_L190_k40_correct += (labels == DenseNet_BC_L190_k40_accuracy.squeeze(0)).sum().item()
    DenseNet_BC_L190_k40_success += ((labels == DenseNet_BC_L190_k40_accuracy.squeeze(0))*(labels != DenseNet_BC_L190_k40_successrate.squeeze(0))).sum().item()
    pyramidnet_correct += (labels == pyramidnet_accuracy.squeeze(0)).sum().item()
    pyramidnet_success += ((labels == pyramidnet_accuracy.squeeze(0))*(labels != pyramidnet_successrate.squeeze(0))).sum().item()
    gdas_correct += (labels == gdas_accuracy.squeeze(0)).sum().item()
    gdas_success += ((labels == gdas_accuracy.squeeze(0))*(labels != gdas_successrate.squeeze(0))).sum().item()
    vgg_correct += (labels == vgg_accuracy.squeeze(0)).sum().item()
    vgg_success += ((labels == vgg_accuracy.squeeze(0))*(labels != vgg_successrate.squeeze(0))).sum().item()
    incv3_correct += (labels == incv3_accuracy.squeeze(0)).sum().item()
    incv3_success += ((labels == incv3_accuracy.squeeze(0))*(labels != incv3_successrate.squeeze(0))).sum().item()
    mobilv2_correct += (labels == mobilv2_accuracy.squeeze(0)).sum().item()
    mobilv2_success += ((labels == mobilv2_accuracy.squeeze(0))*(labels != mobilv2_successrate.squeeze(0))).sum().item()
    resnet18_correct += (labels == resnet18_accuracy.squeeze(0)).sum().item()
    resnet18_success += ((labels == resnet18_accuracy.squeeze(0))*(labels != resnet18_successrate.squeeze(0))).sum().item()

def get_success_rate(correct, success, total):
    if total > 0:
        success_rate = 100.0 * success / total
        accuracy = 100.0 * correct / total
        return accuracy, success_rate
    else:
        return 0, 0

model_performance = {
    'vgg19_bn': (vgg_correct, vgg_success, img_num),
    'WRN': (WRN_correct, WRN_success, img_num),
    'ResNeXt_29_8_64d': (ResNeXt_29_8_64d_correct, ResNeXt_29_8_64d_success, img_num),
    'DenseNet_BC_L190_k40': (DenseNet_BC_L190_k40_correct, DenseNet_BC_L190_k40_success, img_num),
    'pyramidnet': (pyramidnet_correct, pyramidnet_success, img_num),
    'gdas': (gdas_correct, gdas_success, img_num),
    'inception_v3': (incv3_correct, incv3_success, img_num),
    'mobilenet_v2': (mobilv2_correct, mobilv2_success, img_num),
    'resnet18': (resnet18_correct, resnet18_success, img_num)
}

for model_name, metrics in model_performance.items():
    accuracy, success_rate = get_success_rate(*metrics)
    logging.info(f"{model_name}: Accuracy = {accuracy:.2f}%, Attack Success Rate = {success_rate:.2f}%")
    print(f"{model_name}: Accuracy = {accuracy:.2f}%, Attack Success Rate = {success_rate:.2f}%")