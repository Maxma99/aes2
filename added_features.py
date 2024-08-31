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




def predict_chunk(train: pd.DataFrame) -> pd.DataFrame:

    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")


    sentence_encoder = hub.KerasLayer(embed)


    # train
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




# %%
if __name__ == "__main__":
    submission_1 = predict_chunk(train)
    #submission.to_pickle('/home/mcq/GitHub/aes2/train_data/argument-feat.pkl')
    #submission.head(3)



