import torch
from torch import nn
import torch_geometric

from denas_score.utils import *
import sys
sys.path.append('..')
from graphgym.config import cfg

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for m in net.children():
    # for layer in net.modules():
    #     if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
    #         # what is this? 
    #         continue
        if isinstance(m, (nn.Conv2d, nn.Linear, torch_geometric.nn.dense.linear.Linear, nn.modules.sparse.Embedding)):
            metric_array.append(metric(m))
        else:
            metric_array.extend(get_layer_metric_array(m, metric, mode))

    return metric_array



def compute_synflow_per_weight(net, batch, mode, dtype):
    device = cfg.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            # state_dict return all the parameters including both learnable and unlearnable
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    # net.double()
    # input_dim = list(batch[0, :].shape)
    # batch = torch.ones([1] + input_dim).double().to(device)  # batch size = 1
    batch.x = torch.ones(batch.x.shape, dtype=dtype).to(device)

    output, true, last_hidden = net(batch)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs

def syncflow_score(loaders, model, loader_size=1, dtype=torch.int64):
    model.train()
    model.requires_grad_(True)
    model.zero_grad()

    loader = loaders[0]  # training data

    network_weight_gaussian_init(model)

    # create input 
    # input_list = []
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
        
        # input_list.append(random1)
    # input = torch.cat(input_list, dim=0)
    
        batch.x = random1  # must use batch inside loop, if outside, batch becomes in cpu device
        grads_abs_list = compute_synflow_per_weight(net=model, batch=batch, mode='', dtype=dtype)
    
    
    score = 0
    for grad_abs in grads_abs_list:
        score += float(torch.mean(torch.sum(grad_abs, dim=list(range(len(grad_abs.shape))))))
        # if len(grad_abs.shape) == 4:
        #     score += float(torch.mean(torch.sum(grad_abs, dim=[1,2,3])))
        # elif len(grad_abs.shape) == 2:
        #     score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        # else:
        #     raise RuntimeError('!!!')


    return -1 * score