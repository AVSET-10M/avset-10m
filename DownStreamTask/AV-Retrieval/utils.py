from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from pathlib import Path
import einops
import math

import torch
import glob
import os
import time

import seaborn as sns


from metric import Retrieval_metrics

from loss import cross_entropy, get_CLIP_loss, get_L2_loss

def cal_sim_norm(audio_embs, text_embs, temp):
    audio_embs = F.normalize(audio_embs,dim=1)
    text_embs = F.normalize(text_embs,dim=1)
    sim = (audio_embs @ text_embs.T) / temp
    # sim = torch.softmax(sim,dim=1)
    # sim = torch.flatten(sim) 
    return sim

def get_label_heatmap(name, audio_embs, text_embs, temp):
    audio_embs = F.normalize(audio_embs,dim=1)
    text_embs = F.normalize(text_embs,dim=1)
    sim = (audio_embs @ text_embs.T) / temp
    sim = torch.softmax(sim,dim=1)
    torch.save(sim, name+ '.pt')
    heatmap = sns.heatmap(data=sim.cpu().numpy(),vmin=0, vmax=1,cbar=False,square=True,annot=False,cmap="RdBu_r")
    heatmap.get_figure().savefig(name+'.png')
    return

def get_uniform_ball_noise(input_shape, radius=0.1):
    uniform_noise_ball = torch.randn(input_shape)  # normal distribution
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=-1)
    u = torch.rand(input_shape[0])  # unified distribution
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
    return uniform_noise_ball

def noise_injection(x, variance=0.001, modality_offset=None, uniform_noise=False):
    device = x.device
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, radius=std).to(device)
    else:
        x = x + (torch.randn(x.shape).to(device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=-1)

def gather_tensor(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())] # type: ignore
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False) # type: ignore

    output = torch.cat(tensors_gather, dim=0)
    return output

def gather_infoNCE_loss(video_embs, audio_embs, batch, temp):
    # Einstein sum is more intuitive
    # video_embs = einops.rearrange(video_embs, 'b n c -> (b n) c')
    # audio_embs = einops.rearrange(audio_embs, 'b n c -> (b n) c')
    all_video_embs = gather_tensor(video_embs)
    all_audio_embs = gather_tensor(audio_embs)

    b = batch
    one_label = torch.eye(b).cuda()
    target = torch.zeros(b, b * 4).cuda()
    target[:, b * torch.distributed.get_rank(): b * (torch.distributed.get_rank() + 1)] = one_label # type: ignore

    v2a_logit = torch.einsum('nd,td->nt', [video_embs, all_audio_embs]) / temp
    a2v_logit = torch.einsum('nd,td->nt', [audio_embs, all_video_embs]) / temp

    ce_loss = torch.nn.CrossEntropyLoss()
    loss = ce_loss(v2a_logit, target) + ce_loss(a2v_logit, target)
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def synchronize(self):
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct
    
def get_score_split_sum(mat, split_size):
    left_part, right_part = torch.split(mat, split_size, dim=1)
    # 使用 torch.sum 对每一行进行求和
    sum_left = torch.sum(left_part, dim=1, keepdim=True)
    sum_right = torch.sum(right_part, dim=1, keepdim=True)

    # 使用 torch.cat 将两个部分连接起来
    result = torch.cat([sum_left, sum_right], dim=1)
    # print(result.shape)
    return result

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"path '{directory_path}' create successfully")
        except OSError as e:
            print(f"creating '{directory_path}' but got error: {e}")
    else:
        print(f"path '{directory_path}' already exists")