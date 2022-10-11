import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import sys
sys.path.append('..')
from graphgym.config import cfg

def cross_entropy(pred, true):
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # multiclass
    if pred.ndim > 1 and true.ndim == 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true)
    # binary or multilabel
    else:
        true = true.float()
        return bce_loss(pred, true)


def grad_norm_score(model, loaders, dtype, loader_size=16):
    loader = loaders[0]  # training dataset

    model.train()
    model.requires_grad_(True)
    model.zero_grad()

    network_weight_gaussian_init(model)

    output_list = []
    for idx, batch in enumerate(loader):
        if idx >= loader_size:  # why more loader size is bad??
            break
        batch.split = 'train'
        batch.to(torch.device(cfg.device))

        if dtype == torch.float32:
            random1 = torch.randn(size=batch.x.shape, device=cfg.device, dtype=dtype)
        elif dtype == torch.int64:
            random1 = torch.randint(0, 1, batch.x.shape, device=cfg.device)
            for i in range(batch.x.shape[1]):
                min_int, max_int = torch.min(batch.x[:, i]), torch.max(batch.x[:, i])
                random1[:, i] = torch.randint(min_int, max_int + 1, (1, batch.x.shape[0]), device=cfg.device)
        else:
            raise ValueError('dtype setting error')

        batch.x = random1
        pred, true, last_hidden1 = model(batch)
        output = pred
        output_list.append(pred)

    output = torch.cat(output_list, dim=0)

    # y_true = torch.rand(size=[batch_size, output.shape[1]], device=torch.device('cuda:{}'.format(gpu))) + 1e-10
    # y_true = y_true / torch.sum(y_true, dim=1, keepdim=True)

    num_classes = output.shape[1]
    y = torch.randint(low=0, high=num_classes, size=[output.shape[0]], device=cfg.device)
    loss = cross_entropy(output, y)
    loss.backward()

    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm
