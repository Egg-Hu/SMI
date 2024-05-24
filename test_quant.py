import argparse
import shutil
import time
import os
import sys
import random
import numpy as np
from quantization import *
from synthesis import SMI
from utils import *
from utils.data_utils import find_non_zero_patches
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser(description="SMI", add_help=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", default="deit_base_16_imagenet",
                        help="deit_base_16_imagenet/deit_tiny_16_imagenet")
    parser.add_argument('--dataset', default="/path/to/dataset",
                        help='path to dataset')
    parser.add_argument('--datapool', default="/path/to/datapool",
                        help='path to datapool')
    parser.add_argument("--calib-batchsize", default=32,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--mode", default=0,
                        type=int, help="mode of calibration data, 0: inversion, 1: Gaussian noise")
    parser.add_argument('--w_bit', default=8,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bit', default=8,
                        type=int, help='bit-precision of activation')
    parser.add_argument('--prune_it', nargs='+', type=int, help='the iteration indexes for inversion stopping; -1: to densely invert data; t1 t2 ... tn: to sparsely invert data and perform inversion stopping at t1, t2, ..., tn')
    parser.add_argument('--prune_ratio', nargs='+', type=float, help='the proportion of patches to be pruned relative to the current remaining patches; 0: to densely invert data; r1 r2 ... rn: progressively stopping the inversion of a fraction (r1, r2, ..., rn)$$ of patches at iterations (t1, t2, ..., tn), respectively')

    return parser
def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class Config:
    def __init__(self, w_bit, a_bit):
        self.weight_bit = w_bit
        self.activation_bit = a_bit

def get_teacher(name):
    teacher_name = {'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
                    'deit_base_16_imagenet': 'deit_base_patch16_224',
                    }
    if args.model.split("_")[-1]=='imagenet':
        teacher=build_model(teacher_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    return teacher
def get_student(name):
    model_zoo = {'deit_tiny_16_imagenet': deit_tiny_patch16_224,
                 'deit_base_16_imagenet': deit_base_patch16_224,
                 }
    print('Model: %s' % model_zoo[name].__name__)
    return model_zoo[name]


def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def main():
    print(args)
    seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load bit-config
    cfg = Config(args.w_bit, args.a_bit)

    # Build model
    model = get_student(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model.eval()
    teacher=get_teacher(args.model)
    teacher=teacher.to(device)
    teacher.eval()

    # Build dataloader
    train_loader, val_loader,num_classes,train_transform,_,normalizer = build_dataset(args.model.split("_")[0],args.model.split("_")[-1],args.calib_batchsize,train_aug=False,keep_zero=True,train_inverse=True,dataset_path=args.dataset)


    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Get calibration set
    # Case 0: inversion
    if args.mode == 0:
        iterations=4000#total number of iterations for inversion
        lr_g=0.25#learning rate for inversion
        #coefficient for inversion
        adv=0 #coefficient of adversarial regularization, we do not use it in our work
        bn=0.0 #coefficient of batch normalization regularization, dose not apply to ViTs due to the absence of batch normalization, we borrow a CNN to only facilitate visualization (refer to GradViT: Gradient Inversion of Vision Transformers)
        oh=1 #coefficient of classification loss
        tv1=0#coefficient of total variance regularization with l1 norm, we do not use it in our work
        tv2=0.0001#coefficient of total variance regularization with l2 norm
        l2=0#coefficient of l2 norm regularization, we do not use it in our work
        prune_it = args.prune_it
        prune_ratio = args.prune_ratio
        patch_size=16 if '16' in args.model else 32
        patch_num=197 if patch_size==16 else 50
        img_tag = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-quantization{}{}-smi".format(iterations, lr_g, adv, bn, oh, tv1, tv2, l2, str(prune_it), str(prune_ratio),args.w_bit,args.a_bit)
        datapool_path=os.path.join(args.datapool,'%s/%s'%(args.model,img_tag))#the path to store inverted data
        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print('remove')
        synthesizer = SMI(
            teacher=teacher,teacher_name=args.model, student=model, num_classes=num_classes,
            img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size,lr_g=lr_g,
            synthesis_batch_size=32, sample_batch_size=args.calib_batchsize,
            adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
            save_dir=datapool_path, transform=train_transform,
            normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None)

        print("Generating data...")
        #smi
        _ = synthesizer.synthesize(num_patches=patch_num,prune_it=prune_it,prune_ratio=prune_ratio)
        calibrate_data = synthesizer.sample()
        calibrate_data = calibrate_data.to(device)
        print("Calibrating with generated data...")
        model.model_unfreeze()
        with torch.no_grad():
            _ = model(calibrate_data,current_abs_index=torch.arange(patch_num).repeat(calibrate_data.shape[0], 1).to(calibrate_data.device), next_relative_index=torch.cat([torch.zeros(calibrate_data.shape[0], 1, dtype=torch.long).to(calibrate_data.device),find_non_zero_patches(images=calibrate_data, patch_size=patch_size)], dim=1))

        model.model_quant()
        model.model_freeze()
        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device
        )


    # Case 1: Gaussian noise
    elif args.mode == 1:
        calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)
        print("Calibrating with Gaussian noise...")
        with torch.no_grad():
            output = model(calibrate_data)
        # Freeze model
        model.model_quant()
        model.model_freeze()

        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device
        )
    # Not implemented
    else:
        raise NotImplementedError




def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SMI_test_quant', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
