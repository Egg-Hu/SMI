import os
import torch
import math

import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from synthesis._utils import Normalizer

NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'imagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cub200': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_dogs': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_cars': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_64x64': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'svhn': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'tiny_imagenet': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'imagenet_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # for semantic segmentation
    'camvid': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'nyuv2': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def normalize(tensor, mean, std, reverse=False,keep_zero=True):
    if tensor.dim() not in (3, 4):
        raise ValueError("The input tensor must have 3 or 4 dimensions")

    if keep_zero:
        zero_mask = tensor == 0

    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:
        _mean = _mean[None, :, None, None]
        _std = _std[None, :, None, None]
    else:
        _mean = _mean[:, None, None]
        _std = _std[:, None, None]

    tensor = (tensor - _mean) / _std

    if keep_zero:
        tensor[zero_mask] = 0

    return tensor

class Normalizer(object):
    def __init__(self, mean, std,keep_zero=True):
        self.mean = mean
        self.std = std
        self.keep_zero=keep_zero

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse,keep_zero=self.keep_zero)


def build_dataset(model_type,dataset_type,calib_batchsize=32,train_aug=False,keep_zero=False,train_inverse=False,dataset_path=""):
    if model_type == "deit":
        crop_pct = 0.875
    elif model_type == 'vit':
        crop_pct = 0.9
    elif model_type == 'swin':
        crop_pct = 0.9
    else:
        raise NotImplementedError
    mean,std=NORMALIZE_DICT[dataset_type]['mean'],NORMALIZE_DICT[dataset_type]['std']

    train_transform = build_transform(input_size=224, interpolation="bicubic",mean=mean, std=std, crop_pct=crop_pct,aug=train_aug,keep_zero=keep_zero,inverse_img=train_inverse)
    val_transform = build_transform(input_size=224, interpolation="bicubic",mean=mean, std=std, crop_pct=crop_pct,aug=False,keep_zero=keep_zero,inverse_img=False)
    normalizer=Normalizer(**NORMALIZE_DICT[dataset_type])
    # Data
    if dataset_type=='cifar10':
        train_dataset=torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True,transform=train_transform)
        val_dataset=torchvision.datasets.CIFAR10(root=dataset_path, train=False,download=False, transform=val_transform)
        num_classes=10
    elif dataset_type=='cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=True,download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=False,download=False, transform=val_transform)
        num_classes = 100
    elif dataset_type=='imagenet':
        val_dataset = torchvision.datasets.ImageNet(root=dataset_path, split='val', transform=val_transform)
        num_classes = 1000
    else:
        raise NotImplementedError
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    if dataset_type!='imagenet':
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=calib_batchsize,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader=None
    
    return train_loader, val_loader,num_classes,train_transform,val_transform,normalizer


def build_transform(input_size=224, interpolation="bicubic",
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                    crop_pct=0.875,aug=False,keep_zero=False,inverse_img=False):
    def _pil_interp(method):
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            return Image.BILINEAR
    resize_im = input_size > 32
    t = []
    if resize_im:
        if inverse_img==False:
            size = int(math.floor(input_size / crop_pct))
            ip = _pil_interp(interpolation)
            t.append(
                transforms.Resize(
                    size, interpolation=ip
                ),  # to maintain same ratio w.r.t. 224 images
            )
        t.append(transforms.CenterCrop(input_size))
        if aug:
            t.append(transforms.RandomHorizontalFlip())

    t.append(transforms.ToTensor())
    t.append(Normalizer(mean, std,keep_zero))
    return transforms.Compose(t)

def find_non_zero_patches(images, patch_size):
    bs, c, h, w = images.shape
    patch_h, patch_w = patch_size, patch_size
    if h % patch_h != 0 or w % patch_w != 0:
        raise ValueError("Image dimensions are not divisible by patch size")

    images_reshaped = images.reshape(bs, c, h // patch_h, patch_h, w // patch_w, patch_w)

    images_transposed = images_reshaped.permute(0, 2, 4, 1, 3, 5)

    images_patches = images_transposed.reshape(bs, -1, c * patch_h * patch_w)

    non_zero_patches = torch.any(images_patches != 0, dim=2)

    non_zero_indices = [torch.nonzero(non_zero_patches[i], as_tuple=False).squeeze() + 1 for i in range(bs)]
    non_zero_indices=torch.stack(non_zero_indices,dim=0)
    return non_zero_indices