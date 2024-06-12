#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/datasets/cdeotte/deberta-v3-small-finetuned-v1## ℹ️ Info
# 
# | Method | LB |
# | --- | :---: |
# | LGBM 5 fold | 0.801 |
# | LGBM 15 fold | 0.802 |
# | LGBM 15 fold + post-processing | 0.803 |
# | LGBM 15 fold + 5 fold Deberta (Ensemble) | 0.807 |
# | LGBM 16 fold + 5 fold Deberta (Ensemble) | 0.808 |
# | LGBM 15 fold + 5 fold Deberta (Ensemble) + CountVectorizer | 0.810 |
# | LGBM 15 fold + 5 fold Deberta (as features) + CountVectorizer | 0.811 |
# | LGBM 16 fold + 5 fold Deberta (as features) + CountVectorizer + HashingVectorizer | 0.811 |
# | LGBM 15 fold + 5 fold Deberta (as features) + TfidfVectorizer(ngram(3,6)) + CountVectorizer | 0.812 |
# | LGBM 15 fold + (new)5 fold Deberta (as features) + TfidfVectorizer(ngram(3,6)) + CountVectorizer(ngram(3,5)) | 0.816 |
# | LGBM 15 fold + (new)5 fold Deberta (as features) + TfidfVectorizer(ngram(3,6)) + CountVectorizer(ngram(3,4)) | 0.817 |
# | LGBM 15 fold + (new)5 fold Deberta (as features) + more feature engineering + feature selection | 0.817 |
# 
# 
# * 2024/04/15 : forked original great work kernels
#     * https://www.kaggle.com/code/olyatsimboy/5-fold-deberta-lgbm
#     * https://www.kaggle.com/code/aikhmelnytskyy/quick-start-lgbm
#     * https://www.kaggle.com/code/hideyukizushi/aes2-5folddeberta-lgbm-countvectorizer-lb-810
#     * https://www.kaggle.com/code/olyatsimboy/81-1-aes2-5folddeberta-lgbm-stacking  
#     
#     
# * 2024/04/16 : ~~add HashingVectorizer~~ (not work)
# * 2024/04/21 : Add MetaFEs. Train deberta-v3-large local (5Fold SKF) : https://www.kaggle.com/datasets/hideyukizushi/aes2-400-20240419134941    
# * 2024/04/22 : change TfidfVectorizer ngram to (3,6), CountVectorizer ngram to (3,5)
# * 2024/04/23 : change CountVectorizer ngram to (3,4)
# * 2024/04/24 : MORE FEATURE ENGINEERING + FEATURE SELECTION : https://www.kaggle.com/code/xianhellg/more-feature-engineering-feature-selection-0-817
# ---

# In[ ]:


from click import argument
import pandas as pd
import gc
import pickle
import torch


_test = pd.read_csv("kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
# ENABLE_DONT_WASTE_YOUR_RUN_TIME = len(_test) < 10
ENABLE_DONT_WASTE_YOUR_RUN_TIME = False
if ENABLE_DONT_WASTE_YOUR_RUN_TIME:
    import shutil

#     shutil.copyfile("kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv", "submission.csv")
#     exit(0)
    del _test
    gc.collect()

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
print(f"{CUDA_AVAILABLE = }")


# In[ ]:


# !cp kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv submission.csv


# # <div style="color:white;display:fill;border-radius:5px;background-color:seaGreen;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">▶️ 5 Fold Deberta ◀️</div>

# 

# In[ ]:





# In[ ]:


import xgboost as xgb
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

MAX_LENGTH = 1024
TEST_DATA_PATH = "kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = 'kaggle/input/aes2-400-20240419134941/*/*'
EVAL_BATCH_SIZE = 1


# # Deberta Model

# In[ ]:


def get_deberta_predicted_score(df_test = None):
    models = glob(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(models[0])

    def tokenize(sample):
        return tokenizer(sample['full_text'], max_length=MAX_LENGTH, truncation=True)
    if df_test is None:
        df_test = pd.read_csv(TEST_DATA_PATH)
    ds = Dataset.from_pandas(df_test).map(tokenize).remove_columns(['essay_id', 'full_text'])

    args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=EVAL_BATCH_SIZE, 
        report_to="none"
    )

    predictions = []
    for model in models:
        model = AutoModelForSequenceClassification.from_pretrained(model)
        trainer = Trainer(
            model=model, 
            args=args, 
            data_collator=DataCollatorWithPadding(tokenizer), 
            tokenizer=tokenizer
        )

        preds = trainer.predict(ds).predictions
        predictions.append(softmax(preds, axis=-1))  
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    predicted_score = 0.

    for p in predictions:
        predicted_score += p

    predicted_score /= len(predictions)
    df_test['score'] = predicted_score.argmax(-1) + 1
    df_test.head()
    df_test[['essay_id', 'score']].to_csv('submission1.csv', index=False)
    return predicted_score


# # <div style="color:white;display:fill;border-radius:5px;background-color:seaGreen;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">▶️ Prize feedback ◀️</div>

# In[ ]:


def get_fb3_predicted(df_test = None):
    import pandas as pd
    import fb3_deberta_family_inference_9_28_updated

    if df_test is None:
        df_test = pd.read_csv(TEST_DATA_PATH)
    # if len(df_test) < 10:
    #     fb3_predicted = fb3_deberta_family_inference_9_28_updated.predict_chunk(
    #         df_test.rename(columns={"essay_id": "text_id"})
    #     )
    # else:
    fb3_predicted = fb3_deberta_family_inference_9_28_updated.predict(
        df_test.rename(columns={"essay_id": "text_id"})
    )
    return fb3_predicted







# # <div style="color:white;display:fill;border-radius:5px;background-color:seaGreen;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">▶️ 15 fold LGBM ◀️</div>

# In[ ]:


# Importing necessary libraries
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


# # Features engineering

# In[ ]:


def count_spelling_errors(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_.lower() for token in doc]
    spelling_errors = sum(1 for token in lemmatized_tokens if token not in english_vocab)
    return spelling_errors

def removeHTML(x):
    html=re.compile(r'<.*?>')
    return html.sub(r'',x)
def dataPreprocessing(x):
    # Convert words to lowercase
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
    # Remove empty characters at the beginning and end
    x = x.strip()
    return x


# ## Paragraph Features

# In[ ]:


# paragraph features
def remove_punctuation(text):
    """
    Remove all punctuation from the input text.
    
    Args:
    - text (str): The input text.
    
    Returns:
    - str: The text with punctuation removed.
    """
    # string.punctuation
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def Paragraph_Preprocess(tmp):
    # Expand the paragraph list into several lines of data
    tmp = tmp.explode('paragraph')
    # Paragraph preprocessing
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(dataPreprocessing))
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(remove_punctuation).alias('paragraph_no_pinctuation'))
    tmp = tmp.with_columns(pl.col('paragraph_no_pinctuation').map_elements(count_spelling_errors).alias("paragraph_error_num"))
    # Calculate the length of each paragraph
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len"))
    # Calculate the number of sentences and words in each paragraph
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias("paragraph_sentence_cnt"),
                    pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias("paragraph_word_cnt"),)
    return tmp
# feature_eng

def Paragraph_Eng(train_tmp):
    num_list = [0, 50,75,100,125,150,175,200,250,300,350,400,500,600]
    num_list2 = [0, 50,75,100,125,150,175,200,250,300,350,400,500,600,700]
    aggs = [
        # Count the number of paragraph lengths greater than and less than the i-value
        *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_{i}_cnt") for i in [0, 50,75,100,125,150,175,200,250,300,350,400,500,600,700] ], 
        *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_{i}_cnt") for i in [25,49]], 
        # other
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea2],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea2],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea2],
        *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in paragraph_fea2],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea2],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea2],
        *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in paragraph_fea2],
        *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in paragraph_fea2],  
        *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in paragraph_fea2],  
        ]
    
    df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


# ## Sentence Features

# In[ ]:


# sentence feature
def Sentence_Preprocess(tmp):
    # Preprocess full_text and use periods to segment sentences in the text
    tmp = tmp.with_columns(pl.col('full_text').map_elements(dataPreprocessing).str.split(by=".").alias("sentence"))
    tmp = tmp.explode('sentence')
    # Calculate the length of a sentence
    tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x)).alias("sentence_len"))
    # Filter out the portion of data with a sentence length greater than 15
    tmp = tmp.filter(pl.col('sentence_len')>=15)
    # Count the number of words in each sentence
    tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' '))).alias("sentence_word_cnt"))
    
    return tmp
# feature_eng
sentence_fea = ['sentence_len','sentence_word_cnt']
def Sentence_Eng(train_tmp):
    aggs = [
        # Count the number of sentences with a length greater than i
        *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in [0,15,50,100,150,200,250,300] ], 
        *[pl.col('sentence').filter(pl.col('sentence_len') <= i).count().alias(f"sentence_<{i}_cnt") for i in [15,50] ], 
        # other
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
        *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in sentence_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
        *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in sentence_fea],
        *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in sentence_fea], 
        *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in sentence_fea], 
        ]
    df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


# ## Word Features

# In[ ]:


# word feature
def Word_Preprocess(tmp):
    # Preprocess full_text and use spaces to separate words from the text
    tmp = tmp.with_columns(pl.col('full_text').map_elements(dataPreprocessing).str.split(by=" ").alias("word"))
    tmp = tmp.explode('word')
    # Calculate the length of each word
    tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x)).alias("word_len"))
    # Delete data with a word length of 0
    tmp = tmp.filter(pl.col('word_len')!=0)
    
    return tmp
# feature_eng
def Word_Eng(train_tmp):
    aggs = [
        # Count the number of words with a length greater than i+1
        *[pl.col('word').filter(pl.col('word_len') >= i+1).count().alias(f"word_{i+1}_cnt") for i in range(15) ], 
        # other
        pl.col('word_len').max().alias(f"word_len_max"),
        pl.col('word_len').mean().alias(f"word_len_mean"),
        pl.col('word_len').std().alias(f"word_len_std"),
        pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
        pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
        pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
        ]
    df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df


# # Set up loss functions

# In[ ]:


# idea from https://www.kaggle.com/code/rsakata/optimize-qwk-by-lgb/notebook#QWK-objective
def quadratic_weighted_kappa(y_true, y_pred):
    if isinstance(y_pred, xgb.QuantileDMatrix):
        # XGB
        y_true, y_pred = y_pred, y_true

        y_true = (y_true.get_label() + a).round()
        y_pred = (y_pred + a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk

    else:
        # For lgb
        y_true = y_true + a
        y_pred = (y_pred + a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk, True
def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess

a = 2.998
b = 1.092


# # Feature Selection

# In[ ]:


def feature_select_wrapper():
    """
    lgm
    :param train
    :param test
    :return
    """
    # Part 1.
    print('feature_select_wrapper...')
    features = feature_names

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fse = pd.Series(0, index=features)
         
    for train_index, test_index in skf.split(X, y_split):

        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]

        model = lgb.LGBMRegressor(
                    objective = qwk_obj,
                    metrics = 'None',
                    learning_rate = 0.05,
                    max_depth = 5,
                    num_leaves = 10,
                    colsample_bytree=0.3,
                    reg_alpha = 0.7,
                    reg_lambda = 0.1,
                    n_estimators=700,
                    random_state=412,
                    extra_trees=True,
                    class_weight='balanced',
                    verbosity = - 1)

        predictor = model.fit(X_train_fold,
                              y_train_fold,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                              eval_metric=quadratic_weighted_kappa,
                              callbacks=callbacks)
        models.append(predictor)
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + a
        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)
        f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

#         cm = confusion_matrix(y_test_fold_int, predictions_fold, labels=[x for x in range(1,7)])

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                                       display_labels=[x for x in range(1,7)])
#         disp.plot()
#         plt.show()
        print(f'F1 score across fold: {f1_fold}')
        print(f'Cohen kappa score across fold: {kappa_fold}')

        fse += pd.Series(predictor.feature_importances_, features)
#         if ENABLE_DONT_WASTE_YOUR_RUN_TIME:
#             break
    with open("fse.pickle", "wb") as f:
        pickle.dump(fse, f)
    
    # Part 4.
    feature_select = fse.sort_values(ascending=False).index.tolist()[:13000]
    print('done')
    return feature_select


# # Run training pipeline

# In[ ]:


columns = [  
    (
        pl.col("full_text").str.split(by="\n\n").alias("paragraph")
    ),
]
PATH = "kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
paragraph_fea = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt']
paragraph_fea2 = ['paragraph_error_num'] + paragraph_fea

# Load training and testing sets, while using \ n \ n character segmentation to list and renaming to paragraph for full_text data
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)


# TfidfVectorizer parameter
vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(3,6),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
)
vectorizer.fit([i for i in train['full_text']])

vectorizer_cnt = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(2,3),
            min_df=0.10,
            max_df=0.85,
)
vectorizer_cnt.fit([i for i in train['full_text']])

nlp = spacy.load("en_core_web_sm")
with open('kaggle/input/english-word-hx/words.txt', 'r') as file:
    english_vocab = set(word.strip().lower() for word in file)

# Display the first sample data in the training set
train.head(1)


if __name__ == "__main__":
    tmp = Paragraph_Preprocess(train)
    train_feats = Paragraph_Eng(tmp)

    train_feats['score'] = train['score']
    # Obtain feature names
    feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
    print('Features Number: ',len(feature_names))
    train_feats.head(3)

    tmp = Sentence_Preprocess(train)
    train_feats = train_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')

    tmp = Word_Preprocess(train)
    train_feats = train_feats.merge(Word_Eng(tmp), on='essay_id', how='left')


    feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
    print('Features Number: ',len(feature_names))
    train_feats.head(3)


    # TF-IDF
    train_tfid = vectorizer.transform([i for i in train['full_text']])
    # Convert to array
    dense_matrix = train_tfid.toarray()
    # Convert to dataframe
    df = pd.DataFrame(dense_matrix)
    # rename features
    tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = train_feats['essay_id']
    # Merge the newly generated feature data with the previously generated feature data
    train_feats = train_feats.merge(df, on='essay_id', how='left')

    feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
    print('Features Number: ',len(feature_names))
    train_feats.head(3)

    # Count
    train_tfid = vectorizer_cnt.transform([i for i in train['full_text']])
    dense_matrix = train_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [ f'tfid_cnt_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = train_feats['essay_id']
    train_feats = train_feats.merge(df, on='essay_id', how='left')

    feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
    print('Features Number: ',len(feature_names))
    train_feats.head(3)

    # add Deberta predictions to LGBM as features
    deberta_oof = joblib.load('kaggle/input/aes2-400-20240419134941/oof.pkl')
    print(deberta_oof.shape, train_feats.shape)

    for i in range(6):
        train_feats[f'deberta_oof_{i}'] = deberta_oof[:, i]

    feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
    print('Features Number: ', len(feature_names))    

    print(f"{train_feats.shape=}")

    fb_oof = pd.read_csv("kaggle/usr/lib/fb3_deberta_family_inference_9_28_updated/submission.csv")
    fb_oof.head(6)


    train_feats = pd.merge(train_feats, fb_oof, left_on="essay_id", right_on="text_id").drop("text_id", axis=1)
    feature_names += list(fb_oof.columns.drop("text_id"))
    train_feats_argument = pd.read_pickle('kaggle/input/argument-feat.pkl')
    for i in range(2):
        train_feats[f'argument_{i}'] = train_feats_argument.iloc[:, i]
    # Converting the 'text' column to string type and assigning to X
    X = train_feats[feature_names].astype(np.float32).values

    # Converting the 'score' column to integer type and assigning to y
    y_split = train_feats['score'].astype(int).values
    y = train_feats['score'].astype(np.float32).values-a

    f1_scores = []
    kappa_scores = []
    models = []
    predictions = []
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75,first_metric_only=True)]

    if ENABLE_DONT_WASTE_YOUR_RUN_TIME and False:
        with open("kaggle/input/aes2-cache/feature_select.pickle", "rb") as f:
            feature_select = pickle.load(f)
    else:
        feature_select = feature_select_wrapper()

    X = train_feats[feature_select].astype(np.float32).values

    print('Features Select Number: ', len(feature_select))

    with open("train_feats.pickle", "wb") as f:
        pickle.dump(train_feats, f)
    with open("feature_select.pickle", "wb") as f:
        pickle.dump(feature_select, f)
    with open("X.pickle", "wb") as f:
        pickle.dump(X, f)
    with open("y.pickle", "wb") as f:
        pickle.dump(y, f)
    with open("y_split.pickle", "wb") as f:
        pickle.dump(y_split, f)


# # Test Pipeline

# In[ ]:


# if ENABLE_DONT_WASTE_YOUR_RUN_TIME:
#     import shutil

#     shutil.copyfile("kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv", "submission.csv")
def preprocess_test(test: pl.DataFrame| None = None) -> pd.DataFrame:
    
    import argument_classifier
    
    if test is None:
        test = pl.read_csv(PATH + "test.csv").with_columns(columns)
    tmp = Paragraph_Preprocess(test)
    test_feats = Paragraph_Eng(tmp)
    # Sentence
    tmp = Sentence_Preprocess(test)
    test_feats = test_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')
    # Word
    tmp = Word_Preprocess(test)
    test_feats = test_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

    # TfidfVectorizer
    test_tfid = vectorizer.transform([i for i in test['full_text']])
    dense_matrix = test_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = test_feats['essay_id']
    test_feats = test_feats.merge(df, on='essay_id', how='left')

    # CountVectorizer
    test_tfid = vectorizer_cnt.transform([i for i in test['full_text']])
    dense_matrix = test_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [ f'tfid_cnt_{i}' for i in range(len(df.columns))]
    df.columns = tfid_columns
    df['essay_id'] = test_feats['essay_id']
    test_feats = test_feats.merge(df, on='essay_id', how='left')

    # HashingVectorizer
    # test_tfid = vectorizer_hash.transform([i for i in test['full_text']])
    # dense_matrix = test_tfid.toarray()
    # df = pd.DataFrame(dense_matrix)
    # tfid_columns = [ f'tfid_cnt_{i}' for i in range(len(df.columns))]
    # df.columns = tfid_columns
    # df['essay_id'] = test_feats['essay_id']
    # test_feats = test_feats.merge(df, on='essay_id', how='left')
    predicted_score = get_deberta_predicted_score()
    for i in range(6):
        test_feats[f'deberta_oof_{i}'] = predicted_score[:, i]
    fb3_predicted = get_fb3_predicted()
    argument_predicted = argument_classifier.predict_chunk(train = pd.read_csv(TEST_DATA_PATH))
    for i in range(2):
        test_feats[f'argument_{i}'] = argument_predicted.iloc[:, i]
    test_feats = pd.merge(
        test_feats,
        fb3_predicted,
        left_on="essay_id",
        right_on="text_id"
    ).drop("text_id", axis=1)

    # Features number
    feature_names = list(filter(lambda x: x not in ['essay_id','score'], test_feats.columns))
    print('Features number: ',len(feature_names))
    test_feats.head(3)
    return test_feats[feature_select]

def infer(test_feats, models):
    probabilities = []
    for model in models:
        proba = model.predict(test_feats) + a
        probabilities.append(proba)

    # Compute the average probabilities across all models
    predictions = np.mean(probabilities, axis=0)
    predictions = np.round(predictions.clip(1, 6))

    # Print the predictions
    print(predictions)

    submission = pd.read_csv("kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
    submission['score'] = predictions
    submission['score'] = submission['score'].astype(int)
    submission.to_csv("submission.csv", index=None)
    display(submission.head())

if __name__ == "__main__":
    test_feats = preprocess_test()
    print(test_feats.head(5))


# # <div style="color:white;display:fill;border-radius:5px;background-color:seaGreen;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">▶️ Ensemble ◀️</div>

# In[ ]:


# !cat submission.csv

