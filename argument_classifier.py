# %%
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 指定使用 GPU 3
import torch

# %%
import os
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
# import keras

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
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
# env TOKENIZERS_PARALLELISM=false

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
import pandas as pd 
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from datasets import Dataset
from glob import glob
import gc
import torch
from scipy.special import softmax

# %%
import spacy
nlp = spacy.load("en_core_web_sm")
with open('kaggle/input/english-word-hx/words.txt', 'r') as file:
    english_vocab = set(word.strip().lower() for word in file)

# %% [markdown]
# ## 数据load和处理

# %%

# keras.utils.set_random_seed(42) ##每次的随机值相同，方便复现
# keras.mixed_precision.set_global_policy("mixed_float16")



# %%
import polars as pl
columns = [  
    (
        pl.col("full_text").str.split(by="\n\n").alias("paragraph")
    ),
]
PATH = "/home/mcq/GitHub/aes2/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TEST_DATA_PATH = "/home/mcq/GitHub/aes2/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
MAX_LENGTH = 1024
EVAL_BATCH_SIZE = 1


# %% [markdown]
# ## 添加模型

# %%
# Load model directlyfrom transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("/home/mcq/GitHub/aes2/kaggle/input/argument-classifier")
model = AutoModelForSequenceClassification.from_pretrained("/home/mcq/GitHub/aes2/kaggle/input/argument-classifier")

# %%
PATH = '/home/mcq/GitHub/aes2/kaggle/input/learning-agency-lab-automated-essay-scoring-2/'

train = pl.read_csv(PATH + "train.csv").with_columns(columns) 
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

def predict_chunk(train: pd.DataFrame) -> pd.DataFrame:
    def cut(tmp):
        if isinstance(tmp, pd.DataFrame):
            tmp['cut'] = tmp['full_text'].str.slice(0, 513)
        else:
            tmp = tmp.with_columns(pl.col('full_text').str.slice(0, 513).alias("cut"))
        return tmp
    # train = pl.from_pandas(train_pd)

    train = cut(train)

    df =pd.DataFrame({ "cut": train['cut']})

    args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=1, 
        report_to="none"
        
    )
    def tokenize(sample):
        return tokenizer(sample['cut'], max_length=MAX_LENGTH, truncation=True)

    trainer = Trainer(
            model=model, 
            args=args, 
            data_collator=DataCollatorWithPadding(tokenizer), 
            tokenizer=tokenizer
        )
    ds = Dataset.from_pandas(df).map(tokenize).remove_columns(['cut'])
    preds = trainer.predict(ds).predictions

    predictions = pd.DataFrame(softmax(preds, axis=-1))
    # predictions.iloc[:, 0] = predictions.iloc[:, 0] * 5 + 1
    # predictions.iloc[:, 1] = predictions.iloc[:, 1] * (-5) - 1
    torch.cuda.empty_cache()
    gc.collect()
    return predictions

if __name__ == "__main__":
    submission = predict_chunk(train)
    submission.to_pickle('/home/mcq/GitHub/aes2/train_data/argument-feat.pkl')
    submission.head(3)

