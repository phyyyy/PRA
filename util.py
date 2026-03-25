import os
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

from dataset import SelectedImagenet

abbr_full = {
    'wres101': 'wide_resnet101_2.tv2_in1k', 'maevitb': 'vit_base_patch16_224.mae',
    'res18':'resnet18.tv_in1k', 'res34':'resnet34.gluon_in1k', 'res101': 'resnet101.tv2_in1k',
    'vgg19':'vgg19.tv_in1k', 'res152':'resnet152.tv_in1k','seresn101-324d':'seresnext101_32x4d.gluon_in1k',
    "incv3":"inception_v3.tf_in1k", "incv4":"inception_v4.tf_in1k", "incresv2":"inception_resnet_v2.tf_in1k",
    "vitb":"vit_base_patch32_clip_224.laion2b_ft_in12k_in1k", "swinb":"swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "vgg16":"vgg16.tv_in1k", "vgg19bn":"vgg19_bn.tv_in1k", "res50":"resnet50.tv_in1k", "seres50":"seresnet50.ra2_in1k",
    "deit3":"deit3_medium_patch16_224.fb_in22k_ft_in1k", "den121":"densenet121.ra_in1k", "mobv2":"mobilenetv2_120d.ra_in1k",
    "beitb":"beit_base_patch16_224.in22k_ft_in22k_in1k", "mlpmixb":"mixer_b16_224.miil_in21k_ft_in1k", "covnxtb": "convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
    "wdr50":"wide_resnet50_2.tv2_in1k", "advincv3":"inception_v3.tf_adv_in1k", "advincres":"inception_resnet_v2.tf_ens_adv_in1k", "den169": "densenet169.tv_in1k","pnanet":"pnasnet5large.tf_in1k"
}


def get_transforms(data_config, source=True):
    transforms = timm.data.transforms_factory.create_transform(
                        input_size = data_config['input_size'],
                        interpolation = data_config['interpolation'],
                        mean=(0,0,0),
                        std=(1,1,1),
                        crop_pct=data_config['crop_pct'] if not source else 1.,
                        tf_preprocessing=False,
                    )
    if not source:
        transforms.transforms = transforms.transforms[:-2]
    return transforms

def build_dataset(args, data_config):
    img_transform = get_transforms(data_config)
    dataset = SelectedImagenet(imagenet_val_dir=args.data_dir,
                               selected_images_csv=args.data_info_dir,
                               transform=img_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory = True, num_workers=4)
    return data_loader
    
def build_model(model_name):
    timm_root = './huggingface/hub/pytorch_models'

    model = timm.create_model(abbr_full[model_name], pretrained=False)

    pretrained_path = os.path.join(timm_root, abbr_full[model_name]+'.bin')

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
    else:
        print(f"Pretrained weights not found for model {model_name} at {pretrained_path}")

    data_config = model.default_cfg

    model = nn.Sequential(T.Normalize(data_config["mean"], 
                                      data_config["std"]), 
                          model)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()
    return model, data_config
