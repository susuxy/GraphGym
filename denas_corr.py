import os
import argparse
import pickle
import yaml
from tqdm import tqdm
from random import shuffle
from scipy import stats
import math

from run.main_pyg import *
# from graphgym.train_pyg import *
from nas_utils import *
from denas_score.grad_norm_score import grad_norm_score
from denas_score.zen_score import zen_nas
from denas_score.syncflow_score import syncflow_score



def parse_args():
    parser = argparse.ArgumentParser(description='calculate denas correlation')
    parser.add_argument('--model_dict', default='perf_arxiv.pkl', type=str, help='file name of model dictionary')
    parser.add_argument('--model_config', default='config_arxiv', type=str, help='folder saving the model yaml file')
    parser.add_argument('--eval_metric', type=str, default='auc')
    parser.add_argument('--log_name', default='arxiv_node', type=str, help='file name of log')
    parser.add_argument('--repeat_time', default=32, type=int, help='repeat time for calculating scores')
    # parser.add_argument('--input_dtype', default=torch.int64, type=torch.dtype)

    # denas score type
    parser.add_argument('--denas', default='grad_norm', choices=['grad_norm', 'zen_nas', 'syncflow'], type=str)
    # grad norm score
    parser.add_argument('--loader_size', default=1, type=int)
    
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='ss', help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1, help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true', help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='See graphgym/config.py for remaining options.')
    return parser.parse_known_args()[0]

def runner(args):
    # Load config file
    load_cfg(cfg, args)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    auto_select_device()
    # Set machine learning pipeline
    loaders = create_loader()
    model = create_model()

    cfg.params = params_count(model)

    if args.denas == 'zen_nas':
        score = zen_nas(loaders, model, repeat=args.repeat_time, mixup_gamma=1e-2, dtype=args.input_dtype, loader_size=args.loader_size)
    elif args.denas == 'grad_norm':
        score = grad_norm_score(model, loaders, dtype=args.input_dtype, loader_size=args.loader_size)
    elif args.denas == 'syncflow':
        score = syncflow_score(loaders, model, loader_size=args.loader_size, dtype=args.input_dtype)

    return score

args = parse_args()
if 'molhiv' in args.model_dict:
    args.input_dtype = torch.int64
elif 'arxiv' in args.model_dict:
    args.input_dtype = torch.float32


# logger
log_file_path = os.path.join('denas_output', 'log_file', args.log_name + '.log')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_file_path)
logger.addHandler(handler)

with open(log_file_path, 'w') as f:
    pass

model_dict_path = os.path.join('denas_model', args.model_dict)
model_saving_folder = os.path.join('denas_model', args.model_config)

# read the model performance dict
with open(model_dict_path, 'rb') as f:
    model_key_dict = pickle.load(f)

# calculate the denas score & read train score for every model
train_score_list, denas_score_list = [], []
model_config_list = os.listdir(model_saving_folder)
shuffle(model_config_list)
model_number_nan = 0
for idx, model_yaml_name in enumerate(tqdm(model_config_list, desc='read all models')):
    model_yaml_file = os.path.join(model_saving_folder, model_yaml_name)
    with open(model_yaml_file, "r") as f:
        model_config = yaml.safe_load(f)

    # add conditions for calculating denas score
    # if model_config['gnn']['layers_pre_mp'] not in [1,2]:
    #     continue
    # if model_config['gnn']['layers_mp'] not in [1,2]:
    #     continue
    # if model_config['gnn']['layers_post_mp'] not in [1,2]:
    #     continue
    # if model_config['gnn']['layer_type'] not in ['sageconv']:
    #     continue

    # run model 
    args.cfg_file = model_yaml_file
    denas_score = runner(args=args)
    if math.isnan(denas_score):
        model_number_nan += 1
        continue

    # train score
    if seri_dict(order_dict(model_config)) in model_key_dict:
        train_score = model_key_dict[seri_dict(order_dict(model_config))]
    else:
        continue

    train_score_list.append(train_score)
    denas_score_list.append(denas_score)

    spearman_corr, pvalue = stats.spearmanr(train_score_list, denas_score_list)
    logger.info(f"epoch is {idx}")
    logger.info(f"train_score is {train_score:.4f}, denas_score is {denas_score:.4f}")
    logger.info(f"spearman_correlation {spearman_corr:.4f}")
logger.info(f"number of model with nan denas score {model_number_nan}")




