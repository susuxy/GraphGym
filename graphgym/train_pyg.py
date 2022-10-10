import logging
import time

import torch
import torch.nn as nn
import torch_geometric
import numpy as np

from graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from tqdm import tqdm


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



def zen_nas(loaders, model, repeat=32, mixup_gamma=1e-2):
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

