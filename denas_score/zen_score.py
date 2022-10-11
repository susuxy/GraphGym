import torch
import numpy as np
import copy
from tqdm import tqdm

from utils import *
import sys
sys.path.append('..')
from graphgym.config import cfg



def zen_nas(loaders, model, repeat=32, mixup_gamma=1e-2, dtype=torch.float32, loader_size=16):
    loader = loaders[0]  # means training dataset
    nas_score_list = []
    for _ in tqdm(range(repeat), desc='repeat calculate zen-nas score'):
        network_weight_gaussian_init(model)
        all_hidden1, all_hidden2 = [], []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if idx >= loader_size:  # why more loader size is bad??
                    break
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


