# Use GraphGym to calculate DeNas Score

**1. Train the GNN models and get true performance**

Since we don't have a published baselines for DeNas problem, we shall first calculate the true performance of every candidate model. Here we only save the validation accuracy of each model. Run the following codes:

```
python model_perf.py ----model_dict_file_name perf_arxiv.pkl --model_save_folder config_arxiv
```

Here we support two datasets: Arxiv node classification, Molhiv graph classification
model_dict_file_name: base model setting, we use the file as template and update some parameters of it. Now can choose from ['perf_arxiv.pkl', 'perf_molhiv.pkl']
model_save_folder: for each of the trained model, we save the pkl file into this folder. Now can choose from ['config_arxiv', 'config_molhiv']

Users can check the saved model under 'denas_model' folder.


**2. Calculate the DeNas score for every candidate model**

Here we calculate the DeNas score for each candidate model generated from the above procedure. Run the following codes:
```
python denas_corr.py --denas grad_norm
```
We can also calculate the DeNas scores for both arxiv and molhiv dataset now. Users need to change the input parameters based on the parser setting under denas_corr.py file (default setting is for arxiv dataset). Besides, users can choose set --denas for ['grad_norm', 'zen_nas', 'syncflow']. Users can add more denas scores under the folder 'denas_score'.

We may sometimes only care about the DeNas score for some certain circumstances. For example, we want to calculate the spearman correlation between denas score and true score ONLY for sageconv layer. Then, we can add 
```
if model_config['gnn']['layer_type'] not in ['sageconv']:
    continue
```
under the 'for' loop in denas_corr.py file. (I comment it under denas_corr.py)


The results of log files will be saved under 'denas_output/log_file' folder.

**3. Calculate the Spearman Correlation and plot the figure of denas score and true score**
Run the following codes:
```
python corr_plot.py --log_name arxiv_node
```
Specfying the log_name from the previous procedure. The output figures will be shown under 'denas_fig' folder.


