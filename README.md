# Propagation Rectified Attack: On Improving Adversarial Transferability

This repository provides the **official source code** of the paper:

**"Propagation Rectified Attack: On Improving Adversarial Transferability"**  
Xuxiang Sun*, Hongyu Peng*, Gong Cheng, Junwei Han  
Published in *Science China Information Sciences*, 2025  
📄 [[Paper on ScienceDirect](https://link.springer.com/article/10.1007/s11432-024-4542-8)]

## 🚀 Usage

1. Prepare the necessary pretrained models and place them in ckpt folders.
2. Install the packages listed below:
```
pip install torch torchvision timm tqdm numpy pillow pyyaml
```
3. To generate adversarial examples on ImageNet, run the following command:
```
python imagenet/attack_resnet50.py --epsilon 8 --niters 100 --method PRA --batch_size 50 --save_dir adv/PRA --layer_config ./configs/imagenet_resnet50_layers.yaml
```
4. To generate adversarial examples on CIFAR-10, run the following command:
```
python cifar10/attack_vgg19.py --epsilon 4 --niters 100 --method PRA --batch_size 500 --save_dir adv_cifar10/PRA --layer_config ./configs/cifar10_vgg19_layers.yaml
```
5. To evaluate adversarial examples on ImageNet transfer models, run the following command:
```
python test_imagenet.py --dir adv/PRA --log-dir results_imagenet.log
```
6. To evaluate adversarial examples on CIFAR-10 transfer models, run the following command:
```
python test_cifar.py --dir adv_cifar10/PRA --log-dir results_cifar.log
```

## 📌 Citation 
If you find this work useful in your research, please cite:

@article{sun2025propagation,
  title={Propagation rectified attack: on improving adversarial transferability},
  author={Sun, Xuxiang and Peng, Hongyu and Cheng, Gong and Han, Junwei},
  journal={Science China Information Sciences},
  volume={68},
  number={12},
  pages={222102},
  year={2025}
}

## 🧠 Acknowledgement
We thank the authors of the following repositories for making their code open-source.  
1. https://github.com/qizhangli/ILPD-attack
2. https://github.com/CUAI/Intermediate-Level-Attack

## 📬 Contact
If you have any questions, please contact:
Hongyu Peng, hongyupeng@mail.nwpu.edu.cn
