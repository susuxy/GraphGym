import torch
import torch.nn as nn
import torch_geometric
from graphgym.config import cfg

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
            elif isinstance(m, (nn.Linear, torch_geometric.nn.dense.linear.Linear, nn.modules.sparse.Embedding)):
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

