import logging
import time

import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import copy
import torch.nn.functional as F
from tqdm import tqdm


from graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true, last_hidden = model(batch)
        # print(f"pred shape: {pred.shape}. last hidden shape: {last_hidden.shape}")
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true, _ = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler):
    r"""
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


def train_nas(loggers, loaders, model, optimizer, scheduler, patience = 10, metric='auc'):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    counter, best_val_result = 0, -1
    for cur_epoch in tqdm(range(start_epoch, cfg.optim.max_epoch), desc='training the model'):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        # for i in range(1, num_splits):
        i=1  # only for validation set
        eval_epoch(loggers[i], loaders[i], model,
                    split=split_names[i - 1])
        stats = loggers[i].write_epoch(cur_epoch)
        if split_names[i - 1] == 'val':
            val_accuracy = stats[metric]

            # early stopping
            if val_accuracy > best_val_result:
                best_val_result = val_accuracy
                counter = 0 
            else:
                counter += 1
                if counter >= patience:  # perform early stop
                    break
    

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    return best_val_result

def model_last_hidden(loader, model, input_list):
    all_hidden = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch.split = 'train'
            batch.to(torch.device(cfg.device))
            batch.x = input_list[idx]
            pred, true, last_hidden = model(batch)
            last_hidden = last_hidden.detach().cpu()
            all_hidden.append(last_hidden)
    all_hidden = torch.cat(all_hidden, dim=0)
    return all_hidden


def network_weight_gaussian_init(net):
    # child_list = []
    # for m in net.children():
    #     child_list.append(m)
    # if len(child_list) == 0:
    #     print(type(net))
    with torch.no_grad():
        for m in net.children():
            if isinstance(m, nn.Conv2d):
                # print(f"conv: {m}")
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                # print(f"bn: {m}")
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, torch_geometric.nn.dense.linear.Linear)):
                # print(f"linear: {m}")
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.PReLU)):
                # print(f"prelu {m}")
                m.weight.data.fill_(0.25)
            else:
                network_weight_gaussian_init(m)


def gen_params(model):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # if name.split('.')[-1] not in ['bias', 'weight']:
                params.append([name, param])
        return params


def zen_nas(loaders, model, repeat=32, mixup_gamma=1e-2, dtype=torch.float32):
    loader = loaders[0]  # means training dataset
    nas_score_list = []
    for _ in tqdm(range(repeat), desc='repeat calculate zen-nas score'):
        network_weight_gaussian_init(model)
        all_hidden1, all_hidden2 = [], []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                batch.split = 'train'
                batch.to(torch.device(cfg.device))
                batch1 = copy.deepcopy(batch)

                if dtype == torch.float32:
                    random1 = torch.randn(size=batch.x.shape, device=cfg.device, dtype=dtype)
                    random2 = torch.randn(size=batch.x.shape, device=cfg.device, dtype=dtype)
                    random2 = mixup_gamma * random2 + random1
                elif dtype == torch.int64:
                    random1 = torch.randint(0,1, batch.x.shape, device=cfg.device)
                    for i in range(batch.x.shape[1]):
                        min_int, max_int = torch.min(batch.x[:,i]), torch.max(batch.x[:,i])
                        random1[:,i] = torch.randint(min_int, max_int+1, (1, batch.x.shape[0]), device=cfg.device)
                    noise = torch.randint(-1, 2, batch.x.shape, device=cfg.device)
                    random2 = random1 + noise
                    for i in range(batch.x.shape[1]):
                        min_int, max_int = torch.min(batch.x[:, i]), torch.max(batch.x[:, i])
                        random2[:,i][random2[:,i] < min_int] = min_int
                        random2[:,i][random2[:,i] > max_int] = max_int
                    # random1 = batch.x
                    # random2 = batch.x

                else:
                    raise ValueError('dtype setting error')

                batch.x = random1
                pred, true, last_hidden1 = model(batch)
                last_hidden1 = last_hidden1.detach().cpu()
                all_hidden1.append(last_hidden1)

                batch1.x = random2
                pred, true, last_hidden2 = model(batch1)
                last_hidden2 = last_hidden2.detach().cpu()
                all_hidden2.append(last_hidden2)
        all_hidden1 = torch.cat(all_hidden1, dim=0)
        all_hidden2 = torch.cat(all_hidden2, dim=0)

        nas_score = torch.sum(torch.abs(all_hidden1 - all_hidden2), dim=list(range(len(all_hidden1.shape))))
        nas_score = torch.mean(nas_score)

        nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    return avg_nas_score


def zen_nas1(loaders, model, repeat=32, mixup_gamma=1e-2):
    def gen_random_input(loader):
        all_random_input = []
        for batch in loader:
            each_random = torch.randn(size=batch.x.shape, device=cfg.device, dtype=torch.float32)
            all_random_input.append(each_random)
        return all_random_input
    
    loader = loaders[0]  # means training dataset
    nas_score_list = []
    for _ in tqdm(range(repeat), desc='repeat calculate zen-nas score'):
        network_weight_gaussian_init(model)

        input1 = gen_random_input(loader)
        input2 = gen_random_input(loader)
        input2 = [input1[idx] + mixup_gamma * input2[idx] for idx in range(len(input2))]

        embed1 = model_last_hidden(loader, model, input1)
        embed2 = model_last_hidden(loader, model, input2)

        nas_score = torch.sum(torch.abs(embed1 - embed2), dim=list(range(len(embed1.shape))))
        nas_score = torch.mean(nas_score)

        nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    return avg_nas_score




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


def grad_norm_score(model, loaders, dtype, loader_size=64):
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
    y = torch.randint(low=0, high=num_classes, size=[output.shape[0]], device=cfg.device).float()
    loss = cross_entropy(output, y)
    loss.backward()

    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm
