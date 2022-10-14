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
    with torch.no_grad():
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, torch_geometric.nn.dense.linear.Linear, nn.modules.sparse.Embedding)):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.PReLU)):
                m.weight.data.fill_(0.25)
            else:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                if hasattr(m, 'weight_self') and m.weight_self is not None:
                    # generalconv
                    nn.init.normal_(m.weight_self)
                if hasattr(m, 'att_src') and m.att_src is not None:
                    # gatconv
                    nn.init.normal_(m.att_src)
                if hasattr(m, 'att_dst') and m.att_src is not None:
                    # gatconv
                    nn.init.normal_(m.att_dst)


def network_weight_gaussian_init1(net):
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

