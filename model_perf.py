import yaml
import os
import collections
import pickle
import copy
import random
from run.main_pyg import *
from graphgym.train_pyg import train_nas
from tqdm import tqdm
import argparse
from nas_utils import *

# graph
evaluate_metric = 'auc'
sample_yaml = 'example_graph.yaml'
model_dict_file_name = 'perf_molhiv.pkl'
model_save_folder = 'config_molhiv'
patience = 100

# node
# evaluate_metric = 'accuracy'
# sample_yaml = 'example_node.yaml'
# model_dict_file_name = 'perf_arxiv.pkl'
# model_save_folder = 'config_arxiv'
# patience = 200

def parse_args_nas():
   r"""Parses the command line arguments."""
   parser = argparse.ArgumentParser(description='GraphGym')

   parser.add_argument('--cfg',
                       dest='cfg_file',
                       type=str,
                       default='ss',
                       help='The configuration file path.')
   parser.add_argument('--repeat',
                       type=int,
                       default=1,
                       help='The number of repeated jobs.')
   parser.add_argument('--mark_done',
                       action='store_true',
                       help='Mark yaml as done after a job has finished.')
   parser.add_argument('opts',
                       default=None,
                       nargs=argparse.REMAINDER,
                       help='See graphgym/config.py for remaining options.')

   return parser.parse_args()


def runner(model_path):
   args = parse_args_nas()

   args.cfg_file = model_path

   # Load config file
   load_cfg(cfg, args)
   set_out_dir(cfg.out_dir, args.cfg_file)
   # Set Pytorch environment
   torch.set_num_threads(cfg.num_threads)
   dump_cfg(cfg)
   # Repeat for different random seeds
   set_run_dir(cfg.out_dir)
   setup_printing()
   # Set configurations for each run
   cfg.seed = cfg.seed + 1
   seed_everything(cfg.seed)
   auto_select_device()
   # Set machine learning pipeline
   loaders = create_loader()
   loggers = create_logger()
   model = create_model()
   optimizer = create_optimizer(model.parameters())
   scheduler = create_scheduler(optimizer)
   # Print model info
   logging.info(model)
   logging.info(cfg)
   cfg.params = params_count(model)
   logging.info('Num parameters: %s', cfg.params)
   # Start training
   if cfg.train.mode == 'standard':
       val_perf = train_nas(loggers, loaders, model, optimizer, scheduler, metric=evaluate_metric, patience=patience)
   else:
       raise ValueError('train mode is not standard')
   return val_perf

choices = {
   'batchnorm': [True, False],
   'dropout': [0.0, 0.3, 0.6],
   'act': ['relu', 'prelu', 'swish'],
   'agg': ['mean', 'max', 'sum'],
   'stage_type': ['stack', 'skipsum', 'skipconcat'],
   'layers_pre_mp': [1,2,3],
   'layers_mp': [1,2,3],
   'layers_post_mp': [1,2,3]
}


file_path = os.path.join('run', 'configs', 'pyg', sample_yaml)
with open(file_path, "r") as f:
   base_config = yaml.safe_load(f)

order_base_config = order_dict(base_config)

# model_key = seri_dict(order_base_config)

# dict for saving the model and the performance
model_dict_path = os.path.join('denas_model', model_dict_file_name)
if os.path.exists(model_dict_path):
   with open(model_dict_path, 'rb') as f:
       model_key_dict = pickle.load(f)
else:
   model_key_dict = {}
   with open(model_dict_path, 'wb') as f:
       pickle.dump(model_key_dict, f)


copy_base_config = copy.deepcopy(order_base_config)

for _ in tqdm(range(1000), desc='generating models'):
   for key in choices:
       select_value = random.choice(choices[key])
       assert key in copy_base_config['gnn']
       copy_base_config['gnn'][key] = select_value
   copy_model_key = seri_dict(copy_base_config)
   if copy_model_key in model_key_dict:
       continue

   # save yaml file
   copy_base_config_dict = unorder_dict(copy_base_config)
   model_file_path = os.path.join('denas_model', model_save_folder, str(len(model_key_dict)) + '.yaml')
   with open(model_file_path, 'w') as outfile:
       yaml.safe_dump(copy_base_config_dict, outfile)



   # run the model and get accuracy
   val_perf = runner(model_file_path)
   print(f"validation accuracy is {val_perf}")

   # save to dict
   model_key_dict[copy_model_key] = val_perf

   with open(model_dict_path, 'wb') as f:
       pickle.dump(model_key_dict, f)