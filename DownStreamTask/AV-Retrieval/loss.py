import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
# from tllib.modules.kernels import GaussianKernel
bce_withlogits = nn.BCEWithLogitsLoss(reduction='mean')
def get_BCEwithLogits(preds, targets):
    return bce_withlogits(preds, targets)

def get_BCEloss(preds, targets, reduction='mean'):
    bce_loss = nn.BCELoss(reduction=reduction)
    return bce_loss(preds, targets)

nn_ce_loss = nn.CrossEntropyLoss()
def get_nn_CEloss(preds, targets):
    return nn_ce_loss(preds, targets)

loss_domain = torch.nn.NLLLoss()
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_CLIP_loss(CLIP_embs, CLAP_embs, temperature):
    logits = (CLAP_embs @ CLIP_embs.T) / temperature
    targets = torch.eye(logits.shape[0]).to(CLIP_embs.device)
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
    return loss.mean()

def no_softmax_cross_entropy(preds, targets, reduction='none'):
    loss = (-targets * torch.log(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_SoftCLIP_loss(CLIP_embs, CLAP_embs, pre_temperature, soft_temperature, soft_factor):
    device = CLIP_embs.device
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    logits = (CLAP_embs @ CLIP_embs.T) / pre_temperature
    log_logits = F.log_softmax(logits, dim=-1)
    CLIP_similarity = CLIP_embs @ CLIP_embs.T
    CLAP_similarity = CLAP_embs @ CLAP_embs.T

    soft_target = F.softmax(soft_temperature * (CLIP_similarity+CLAP_similarity) / 2, dim=-1).to(device)
    hard_target = torch.eye(logits.shape[0]).to(device)
    target = soft_factor * soft_target + (1 - soft_factor) * hard_target

    hard_loss1 = cross_entropy(logits, hard_target, reduction='none')
    hard_loss2 = cross_entropy(logits.T, hard_target.T, reduction='none')
    hard_loss = (hard_loss1 + hard_loss2) / 2.0  # shape: (batch_size)

    soft_loss1 = kl_loss(log_logits, target)
    soft_loss2 = kl_loss(log_logits.T, target.T)
    soft_loss = (soft_loss1 + soft_loss2) /  2.0
    return soft_loss, hard_loss.mean()

# def get_MMD_loss(source, target):
#     mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
#         kernels=[GaussianKernel(alpha=2 ** k) for k in range(-1, 2)],
#         linear=True
#     )
#     mkmmd_loss.train()
#     loss = mkmmd_loss(source, target)
#     return loss

def get_domain_loss(source, target):
    # return cross_entropy(source, target, reduction='mean')
    return loss_domain(source, target)

def get_L2_loss(source, target):
    bs = 128
    bs_num = source.shape[0] // bs
    dist_list = []
    for i in range(bs_num+1):
        mini_source = source[i*bs:(i+1)*bs, :]
        mini_distance = torch.sum((mini_source.unsqueeze(1) - target.unsqueeze(0)) ** 2, dim=-1)
        dist_list.append(mini_distance)
    L2_distance = torch.cat(dist_list, dim=0)
    return L2_distance.mean()

def get_item_L2_loss(source, target):
    L2_distance = torch.sum((source - target) ** 2, dim=-1)
    return L2_distance.mean()

def get_nn_item_L2_loss(source, target, pos_num, pos_all_one):
    bs = source.shape[0]
    sim = source @ target.T
    nn_value, nn_indexs = torch.topk(sim, pos_num, dim=-1)
    nn_feature = scatter_feature(target, nn_indexs)
    L2_dist = torch.sum((source.unsqueeze(1) - nn_feature) ** 2, dim=-1)
    if pos_all_one:
        positive_value = torch.ones([bs, pos_num]).cuda()
    else:
        positive_value = nn_value
    nn_dist = torch.mean(L2_dist * positive_value, dim=-1)
    return nn_dist.mean()

def scatter_feature(feature, indexs):
    output = []
    indexs = indexs.T
    for index in indexs:
        output.append(feature[index].unsqueeze(1))
    output = torch.cat(output, dim=1)
    return output

def get_info_L2_loss(source, target, pos_num, neg_num, pos_all_one=True):
    bs = source.shape[0]
    sim = source @ target.T
    rsim = -sim
    nn_value, nn_indexs = torch.topk(sim, pos_num, dim=-1)
    fn_indexs = torch.topk(rsim, neg_num, dim=-1).indices
    nn_feature = scatter_feature(target, nn_indexs)
    fn_feature = scatter_feature(target, fn_indexs)
    target_feat = torch.cat([nn_feature, fn_feature], dim=1)

    L2_dist = torch.sum((source.unsqueeze(1) - target_feat) ** 2, dim=-1) / 100
    L2_dist = torch.softmax(L2_dist, dim=-1)
    if pos_all_one:
        positive_value = torch.ones([bs, pos_num]).cuda()
    else:
        positive_value = nn_value
    label = torch.cat([torch.ones([bs, 1]), torch.zeros([bs, neg_num])], dim=-1).cuda()
    nn_dist = torch.sum(L2_dist[:, 0:pos_num] * positive_value, dim=-1)
    dist = torch.cat([nn_dist.unsqueeze(1), L2_dist[:, pos_num:]], dim=-1)

    loss = (label * torch.log(dist)).sum(1)
    return loss.mean()

def get_JSD_loss(source, target):
    return 0

if __name__ == '__main__':
    source = torch.rand(5,2)
    target = torch.rand(5,2)
    print(target)
    loss = get_info_L2_loss(source, target, 3, 2)