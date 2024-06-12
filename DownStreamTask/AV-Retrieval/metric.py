import torch
import torchmetrics
from torchmetrics.retrieval import RetrievalMRR, RetrievalHitRate, RetrievalMAP

def MRR(similarity):
    device = similarity.device
    size1, size2 = similarity.shape
    l = size1 * size2

    target = torch.eye(size1, size2, dtype=torch.bool).to(device)
    indexes = torch.tensor([[i] * size1 for i in range(size2)]).to(device)

    mrr = RetrievalMRR()
    mrr_score = mrr(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l))

    return mrr_score

def Retrieval_metrics(similarity):
    device = similarity.device
    size1, size2 = similarity.shape
    l = size1 * size2

    target = torch.eye(size1, size2, dtype=torch.bool).to(device)
    indexes = torch.tensor([[i] * size1 for i in range(size2)]).to(device)

    metrics = {}

    mrr = RetrievalMRR()
    metrics['mrr'] = mrr(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l)).item()
    # top_1 = RetrievalHitRate(top_k=1)
    # metrics['top_1'] = top_1(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l)).item()
    top_5 = RetrievalHitRate(top_k=5)
    metrics['top_5'] = top_5(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l)).item()
    # top_10 = RetrievalHitRate(top_k=10)
    # metrics['top_10'] = top_10(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l)).item()

    return metrics

def compute_retrieval(similarity_scores):
    gt = torch.arange(0, similarity_scores.shape[0])
    ranks = torch.zeros(similarity_scores.shape[0])

    for index, score in enumerate(similarity_scores):
        inds = torch.argsort(score, descending=True)
        ranks[index] = torch.where(inds == gt[index])[0][0]

    tr1 = 100.0 * len(torch.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(torch.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(torch.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * torch.mean(1/(ranks+1)).item()

    return {'mrr': mrr, "top_1": tr1, "top_5": tr5, "top_10": tr10}

def MiniRetrieval_metrics(similarity):
    device = similarity.device
    size1, size2 = similarity.shape
    l = size1 * size2

    target = torch.eye(size1, size2, dtype=torch.bool).to(device)
    indexes = torch.tensor([[i] * size1 for i in range(size2)]).to(device)

    mrr = RetrievalMRR()
    mrr_score = mrr(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l))
    top_1 = RetrievalHitRate(k=1)
    top_1_score = top_1(similarity.reshape(l), target.reshape(l), indexes=indexes.reshape(l))

    return mrr_score.item(), top_1_score.item()

def gt_compute_retrieval(similarity_scores, gt):
    # gt = torch.arange(0, similarity_scores.shape[0])
    ranks = torch.zeros(similarity_scores.shape[0])

    for index, score in enumerate(similarity_scores):
        inds = torch.argsort(score, descending=True)
        ranks[index] = torch.where(inds == gt[index])[0][0]

    tr1 = 100.0 * len(torch.where(ranks < 1)[0]) / len(ranks)
    tr3 = 100.0 * len(torch.where(ranks < 3)[0]) / len(ranks)
    tr5 = 100.0 * len(torch.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(torch.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * torch.mean(1/(ranks+1)).item()

    return {'mrr': mrr, "top_1": tr1, "top_3": tr3, "top_5": tr5, "top_10": tr10}

def multi_gt_compute_retrieval(similarity_scores, gt):
    # gt = torch.arange(0, similarity_scores.shape[0])
    ranks = torch.zeros(similarity_scores.shape[0])

    for index, score in enumerate(similarity_scores):
        inds = torch.argsort(score, descending=True)
        highest = similarity_scores.shape[1]
        for gt_i in gt[index]:
            temp_rank = torch.where(inds == gt_i)[0][0]
            if temp_rank < highest:
                highest = temp_rank
        ranks[index] = highest

    tr1 = 100.0 * len(torch.where(ranks < 1)[0]) / len(ranks)
    tr3 = 100.0 * len(torch.where(ranks < 3)[0]) / len(ranks)
    tr5 = 100.0 * len(torch.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(torch.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * torch.mean(1/(ranks+1)).item()

    return {'mrr': mrr, "top_1": tr1, "top_3": tr3, "top_5": tr5, "top_10": tr10}