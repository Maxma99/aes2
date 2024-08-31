# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用 GPU 3

# %%
import gc
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import spacy
import string
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.metrics import cohen_kappa_score
from lightgbm import log_evaluation, early_stopping
from sklearn.linear_model import SGDClassifier
import polars as pl
import joblib

import tensorflow_hub as hub
import tensorflow as tf
import statistics
import math


PATH = "kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pd.read_csv(PATH + "train.csv")

embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")


sentence_encoder = hub.KerasLayer(embed)



def predict_chunk(train: pd.DataFrame) -> pd.DataFrame:
    # train
    corpus = train
    column_name = 'full_text'
    sencode_corpus = use_function(corpus, column_name)
    sencode = pd.DataFrame(sencode_corpus)
    # rename features
    sencode_columns = [ f'sencode_{i}' for i in range(len(sencode.columns))]
    sencode.columns = sencode_columns
    # Merge the newly generated feature data with the previously generated feature data
    sencode['essay_id'] = train['essay_id']
    train = train.merge(sencode, on='essay_id', how='left')
    train = train.drop(columns = ['essay_id', 'full_text','score'])

    return train



def predict_chunk_2(train: pd.DataFrame) -> pd.DataFrame:
    #Features engineering
#Preprocessing

    def removeHTML(x):
        html=re.compile(r'<.*?\n>')
        return html.sub(r'',x)

    def dataPreprocessing(x):
        # lowercase
        x = x.lower()
        # Remove HTML
        x = removeHTML(x)
        # Delete strings starting with @
        x = re.sub("@\w+", '',x)
        # Delete Numbers
        x = re.sub("'\d+", '',x)
        x = re.sub("\d+", '',x)
        # Delete URL
        x = re.sub("http\w+", '',x)
        # Replace consecutive empty spaces with a single space character
        x = re.sub(r"\s+", " ", x)
        # Replace consecutive commas and periods with one comma and period character
        x = re.sub(r"\.+", ".", x)
        x = re.sub(r"\,+", ",", x)
        # Delete aposhtroph html
        #x = re.sub(r"\\'", "'", x)
        # Remove empty characters at the beginning and end
        x = x.strip()
        return x

    # Paragraph preprocessing
    train['paragraph_processed'] = [dataPreprocessing(x) for x in train['full_text']]

    # Calculate total number of sentences
    train['sentence_cnt'] = [len(x.split('.')) for x in train['paragraph_processed']]

    # Calculate total number of words
    train['word_cnt'] = [len(x.split(' ')) for x in train['paragraph_processed']]

    def corpus_satistics(data, col, heading_len, split_str, corp_unit):
        corp_unit_len_min = []
        corp_unit_len_max = []
        corp_unit_len_mean = []
        corp_unit_len_median = []
        corp_unit_len_sd = []
        corp_unit_len_quantiles =[]
        
        for z in data[col]:
            corpLen_cnt = []
            for y in z.split(split_str):
                if corp_unit=='word':
                    x=len(y.split(' '))
                    if x>3: # Paragraph heading should be limited to 3 words
                        corpLen_cnt.append(x)
                else:
                    if len(y)>heading_len: # Paragraph heading should be limited to 15-20 characters
                        corpLen_cnt.append(len(y))

            corp_unit_len_min.append(min(corpLen_cnt))
            corp_unit_len_max.append(max(corpLen_cnt))
            corp_unit_len_mean.append(statistics.mean(corpLen_cnt))
            corp_unit_len_median.append(statistics.median(corpLen_cnt))
            if len(corpLen_cnt)>=2: # As some full_texts have just one paragraph
                corp_unit_len_sd.append(statistics.stdev(corpLen_cnt))
                qua = statistics.quantiles(corpLen_cnt, n=10, method='exclusive')
                qua = [0 if i < 0 else i for i in qua]
                corp_unit_len_quantiles.append(qua)
            else:
                corp_unit_len_sd.append(corpLen_cnt[0]) # sd for single paragraph/sentence entries are kept as large 
                corp_unit_len_quantiles.append([0]*9) # quantiles for single paragraph/sentence entries are kept zero



        data[corp_unit + '_len_min'] = corp_unit_len_min
        data[corp_unit + '_len_max'] = corp_unit_len_max
        data[corp_unit + '_len_mean'] = corp_unit_len_mean
        data[corp_unit + '_len_median'] = corp_unit_len_median
        data[corp_unit + '_len_sd'] = corp_unit_len_sd
        data[corp_unit + '_len_qua0'] = [x[0] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua1'] = [x[1] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua2'] = [x[2] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua3'] = [x[3] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua4'] = [x[4] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua5'] = [x[5] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua6'] = [x[6] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua7'] = [x[7] for x in corp_unit_len_quantiles]
        data[corp_unit + '_len_qua8'] = [x[8] for x in corp_unit_len_quantiles]
        return data

        # Statistics for paragraph

    data = train
    col = 'full_text'
    heading_len = 20
    split_str = '\n\n'
    corp_unit = 'paragraph'

    train = corpus_satistics(data, col, heading_len, split_str, corp_unit)

        # Statistics for sentence

    data = train
    col = 'paragraph_processed'
    heading_len = 15
    split_str = '.'
    corp_unit = 'sentence'

    train = corpus_satistics(data, col, heading_len, split_str, corp_unit)

        # Statistics for word

    data = train
    col = 'paragraph_processed'
    #heading_len = 15
    split_str = '.'
    corp_unit = 'word'
    train = corpus_satistics(data, col, heading_len, split_str, corp_unit)

    train = train.drop(columns = ['essay_id', 'full_text','score'])
    return train





# %%
if __name__ == "__main__":
    
    sencode_corpus = []
    def use_function(corpus, column_name):        
        
        for x in corpus[column_name]:
      
            if len(x.split('.'))<2:
                sencode_essay = [0.]*512

            else:
                enc_raw = sentence_encoder(x.split('.'))[:-1]
                sencode_essay = tf.math.reduce_sum(enc_raw, 0).numpy()/math.sqrt(len(x.split('.')))

            sencode_corpus.append(sencode_essay)
        return sencode_corpus


    submission_1 = predict_chunk(train)
    # submission_2 = predict_chunk_2(train)
    submission_1.to_pickle('/home/mcq/GitHub/aes2/train_data/add2-feat.pkl')
    #submission.head(3)



