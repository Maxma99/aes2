#!/usr/bin/env python
# coding: utf-8

# If you have time, please check my other notebooks.
# 
# * Train : https://www.kaggle.com/kojimar/fb3-single-pytorch-model-train
# * Inference : https://www.kaggle.com/kojimar/fb3-single-pytorch-model-inference

# References
# * https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
# * https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference

# In[ ]:


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8"
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # CFG

# Deberta Family ver. 13
# 
# Sep. 28, LB 21th
# 
# * CFG1 : 10 fold deberta-v3-base CV/LB: 0.4595/0.44
# * CFG2 : 10 fold deberta-v3-large CV/LB: 0.4553/0.44
# * CFG3 : 10 fold deberta-v2-xlarge CV/LB: 0.4604/0.44
# * CFG4 : 10 fold deberta-v3-base FGM CV/LB: 0.4590/0.44 ~~~~
# * CFG5 : 10 fold deberta-v3-large FGM CV/LB: 0.4564/0.44
# * CFG6 : 10 fold deberta-v2-xlarge CV/LB: 0.4666/0.44
# * CFG7 : 10 fold deberta-v2-xlarge-mnli CV/LB: 0.4675/0.44 ~~~~
# * CFG8 : 10 fold deberta-v3-large unscale CV/LB: 0.4616/0.43
# * CFG9 : 10 fold deberta-v3-large unscale CV/LB: 0.4548/0.43
# * CFG10 : 10 fold deberta-v3-large unscale CV/LB: 0.4569/0.43

# In[ ]:


# class CFG1:
#     model = "microsoft/deberta-v3-base"
#     path = "../input/0911-deberta-v3-base/"
#     base = "../input/fb3models/microsoft-deberta-v3-base/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=24
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0
    
# class CFG2:
#     model = "microsoft/deberta-v3-large"
#     path = "../input/0911-deberta-v3-large/"
#     base = "../input/fb3models/microsoft-deberta-v3-large/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=16
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0
    
# class CFG3:
#     model = "microsoft/deberta-v2-xlarge"
#     path = "../input/0911-deberta-v2-xlarge/"
#     base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=4
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0

class CFG4:
    model = "microsoft/deberta-v3-base"
    path = "/home/mcq/GitHub/aes2/kaggle/input/fb3models/0913-deberta-v3-base-FGM/"
    base = "/home/mcq/GitHub/aes2/kaggle/input/fb3models/microsoft-deberta-v3-base/"
    config_path = base + "config.json"
    tokenizer = AutoTokenizer.from_pretrained(base)
    gradient_checkpointing=False
    batch_size=24
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=10
    trn_fold=list(range(n_fold))
    num_workers=4
    weight = 1.0
    
# class CFG5:
#     model = "microsoft/deberta-v3-large"
#     path = "../input/0914-deberta-v3-large-fgm/"
#     base = "../input/fb3models/microsoft-deberta-v3-large/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=16
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0
    
# class CFG6:
#     model = "microsoft/deberta-v2-xlarge"
#     path = "../input/0919-deberta-v2-xlarge/"
#     base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=4
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0
    
class CFG7:
    model = "microsoft/deberta-v2-xlarge-mnli"
    path = "/home/mcq/GitHub/aes2/kaggle/input/fb3models/0919-deberta-v2-xlarge-mnli/"
    base = "/home/mcq/GitHub/aes2/kaggle/input/fb3models/microsoft-deberta-v2-xlarge/"
    config_path = base + "config.json"
    tokenizer = AutoTokenizer.from_pretrained(base)
    gradient_checkpointing=False
    batch_size=4
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=10
    trn_fold=list(range(n_fold))
    num_workers=4
    weight = 1.0
    
# class CFG8:
#     model = "microsoft/deberta-v3-large"
#     path = "../input/0925-deberta-v3-large-unscale/"
#     base = "../input/fb3models/microsoft-deberta-v3-large/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=16
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0
    
# class CFG9:
#     model = "microsoft/deberta-v3-large"
#     path = "../input/0926-deberta-v3-large-unscale/"
#     base = "../input/fb3models/microsoft-deberta-v3-large/"
#     config_path = base + "config/config.json"
#     tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
#     gradient_checkpointing=False
#     batch_size=16
#     target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#     seed=42
#     n_fold=10
#     trn_fold=list(range(n_fold))
#     num_workers=4
#     weight = 1.0
    
class CFG10:
    model = "microsoft/deberta-v3-large"
    path = "/home/mcq/GitHub/aes2/kaggle/input/fb3models/0927-deberta-v3-large-unscale/"
    base = "/home/mcq/GitHub/aes2/kaggle/input/fb3models/microsoft-deberta-v3-large/"
    config_path = base + "config.json"
    # tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    tokenizer = AutoTokenizer.from_pretrained(base)
    gradient_checkpointing=False
    batch_size=16
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=10
    trn_fold=list(range(n_fold))
    num_workers=4
    weight = 1.0
    

# # Utils

# In[ ]:


# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

def get_logger(filename='inference'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# # OOF

# In[ ]:


# ====================================================
# oof
# ====================================================


# # Dataset

# In[ ]:


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        #max_length=CFG.max_len,
        #pad_to_max_length=True,
        #truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


# # Model

# In[ ]:


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    
class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings
        

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = AutoConfig.from_pretrained(config_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# # inference

# In[ ]:


# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


# In[ ]:


import os



def predict_chunk(test: pd.DataFrame) -> pd.DataFrame:
    CFG_list = [CFG4, CFG7, CFG10]

    for _idx, CFG in enumerate(CFG_list):
    #     test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
        
        submission = pd.DataFrame(np.ones((len(test), 7)))
        submission.columns = ['text_id','cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar','conventions']
        submission["text_id"] = test["text_id"]
        print(test["text_id"].head(5))
        print(submission["text_id"].head(5))
        # sort by length to speed up inference
        test['tokenize_length'] = [len(CFG.tokenizer(text)['input_ids']) for text in test['full_text'].values]
        test = test.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

        test_dataset = TestDataset(CFG, test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=CFG.batch_size,
                                 shuffle=False,
                                 collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
                                 num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        predictions = []
        for fold in CFG.trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
            print(model)
            print(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                               map_location=torch.device('cpu'))
            model.load_state_dict(state['model'], strict=False)
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            del model, state, prediction; gc.collect()
            torch.cuda.empty_cache()
            break
        predictions = np.mean(predictions, axis=0)
        test[CFG.target_cols] = predictions
        submission = submission.drop(columns=CFG.target_cols).merge(test[['text_id'] + CFG.target_cols], on='text_id', how='left')
    #         display(submission.head())
        submission[['text_id'] + CFG.target_cols].to_csv(f'submission_{_idx + 1}.csv', index=False)
        torch.cuda.empty_cache()
        gc.collect()
        return submission

         
def predict(test: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.DataFrame()
    CFG_list = [CFG4, CFG7, CFG10]
    
    for j, CFG in enumerate(CFG_list):
        result = pd.DataFrame()
        CFG_inli = [CFG]
        oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Model: {CFG.model} Score: {score:<.4f}  Scores: {scores}')
        
        for i, chunk in enumerate(np.array_split(test, 10)):
            if len(chunk) == 0:
                continue
            result = pd.concat(
                [result, predict_chunk(test=chunk.reset_index(drop=True))]
                ).reset_index(drop=True)
        if j > 0:
            result = result.drop(columns=['text_id'])
            result = result.add_suffix(f'_{j}')
    prediction = pd.concat([prediction, result], axis = 1)
    return prediction

if __name__ == "__main__":
    test = pd.read_csv(
            '/home/mcq/GitHub/aes2/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv'
            ).rename(columns={"essay_id": "text_id"})
    submission = predict(test=test.head(500))
    submission.to_csv(f'fb3_feat.csv', index=False)


# %%
