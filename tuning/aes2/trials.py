import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, cohen_kappa_score
# from aes2_added_fb_prize_as_features_preprocessing import *
import nni
import xgboost as xgb
import lightgbm as lgb
from xgboost import DMatrix
import logging
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
LOG = logging.getLogger('aes2')
sys.path.append('/home/mcq/GitHub/aes2/')

import aes2_added_fb_prize_as_features_preprocessing
# %%
import gc

gc.collect()

def get_default_parameters():
    params = {
        'gpu_id': 4,
        'ratio' : 0.749,
        'max_depth_lgb' : 8 ,
        'num_leaves_lgb' : 10 ,
        'reg_alpha_lgb' : 0.7 ,
        'reg_lambda_lgb' : 0.1 ,
        'max_depth_xgb' : 8 ,
        'num_leaves_xgb' : 10 ,
        'reg_alpha_xgb' : 0.1 ,
        'reg_lambda_xgb' : 0.8
    }

    return params

def clean_feature_names(features):
    illegal_chars = ['[', ']', '<', '>']
    cleaned_features = []
    for feature in features:
        for char in illegal_chars:
            feature = feature.replace(char, 'lessthan')
        cleaned_features.append(feature)
    return cleaned_features

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

class Predictor:
    def __init__(self, models: list, ratio):
        self.models = models
        self.ratio = ratio
#         self.xgb_boost_best_iter = models[1].
    def predict(self, X):
        n_models = len(self.models)
        predicted = None
        n = self.ratio
        for i, model in enumerate(self.models):
            if i == 0:
                predicted = n*model.predict(X)
            else:
                # if not isinstance(X, xgb.DMatrix):
                #     X = xgb.DMatrix(X)
                predicted += (1-n)*model.predict(X)
        return predicted






def run_experiment(tuner_params):
    n_splits = int(tuner_params['n_splits'])
    models = []
    predictions = []
    f1_scores = []
    kappa_scores = []
    params = {
        'n_splits' : 15,
        'learning_rate_lgb' : 0.05 ,
        'colsample_bytree_lgb' : 0.3 ,
        'n_estimators_lgb' : 700, 
        'learning_rate_xgb' : 0.1 ,
        'colsample_bytree_xgb' : 0.5 ,
        'n_estimators_xgb' : 1024
    }

    print('......DATA LOADING......')
    import pickle
    with open("/home/mcq/GitHub/aes2/train_data/train_feats.pickle", "rb") as f:
        train_feats = pickle.load(f)
    with open("/home/mcq/GitHub/aes2/train_data/X.pickle", "rb") as f:
        X = pickle.load(f)
    with open("/home/mcq/GitHub/aes2/train_data/y.pickle", "rb") as f:
        y = pickle.load(f)
    with open("/home/mcq/GitHub/aes2/train_data/y_split.pickle", "rb") as f:
        y_split = pickle.load(f)
    with open(
        "/home/mcq/GitHub/aes2/train_data/feature_select.pickle", "rb"
    ) as f:
        feature_select = pickle.load(f)
        
    aes2_added_fb_prize_as_features_preprocessing.feature_select = feature_select
    train_feats.columns = clean_feature_names(train_feats.columns)
    feature_select = clean_feature_names(feature_select)
    X = train_feats[feature_select].astype(np.float32).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(skf.split(X, y_split), 1):
    # Split the data into training and testing sets for this fold
        print('fold',i)
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]
        callbacks = [log_evaluation(period=75), early_stopping(stopping_rounds=75,first_metric_only=True)]
        light = lgb.LGBMRegressor(
                objective = qwk_obj,
                metrics = 'None',
                learning_rate = params['learning_rate_lgb'],
                max_depth = int(tuner_params['max_depth_lgb']),
                num_leaves = int(tuner_params['num_leaves_lgb']),
                colsample_bytree=params['colsample_bytree_lgb'],
                reg_alpha = tuner_params['reg_alpha_lgb'],
                reg_lambda = tuner_params['reg_lambda_lgb'],
                n_estimators=int(params['n_estimators_lgb']),
                feature_fraction=1.0,
                random_state=42,
                extra_trees=True,
                class_weight='balanced',
                # device_type = 'gpu',
                # device='gpu' if CUDA_AVAILABLE else 'cpu',
                verbosity = - 1
            )

        # Fit the model on the training data for this fold  
        light.fit(
            X_train_fold,
            y_train_fold,
            eval_names=['train', 'valid'],
            eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks
        )
        
        
        
        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=75),
            xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
        ]
        xgb_regressor = xgb.XGBRegressor(
            objective = qwk_obj,
            metrics = 'None',
            learning_rate = params['learning_rate_xgb'],
            max_depth = int(tuner_params['max_depth_xgb']),
            num_leaves = int(tuner_params['num_leaves_xgb']),
            colsample_bytree=params['colsample_bytree_xgb'],
            reg_alpha = tuner_params['reg_alpha_xgb'],
            reg_lambda = tuner_params['reg_lambda_xgb'],
            n_estimators=int(params['n_estimators_xgb']),
            random_state=42,
            extra_trees=True,
            class_weight='balanced',
            tree_method="gpu_hist",
            # device="gpu" if CUDA_AVAILABLE else "cpu",
            gpu_id = int(tuner_params['gpu_id'])
        #             device='gpu',
        #             verbosity = 1
        )
        
        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=25),
            xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
        ]
        xgb_regressor.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=xgb_callbacks
        )
        
        
        predictor = Predictor([light, xgb_regressor], tuner_params['ratio'])
        
        models.append(predictor)
        # Make predictions on the test data for this fold
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + a
        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)
        # Calculate and store the F1 score for this fold
        f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        # Calculate and store the Cohen's kappa score for this fold
        kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
        nni.report_intermediate_result(kappa_fold)
        kappa_scores.append(kappa_fold)

        gc.collect()

    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)
    # Print the mean scores
    print(f'Mean F1 score across {n_splits} folds: {mean_f1_score}')
    print(f'Mean Cohen kappa score across {n_splits} folds: {mean_kappa_score}')
    nni.report_final_result(mean_kappa_score)


if __name__ == '__main__':
    
    a = 2.998
    b = 1.092
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        run_experiment(tuner_params=RECEIVED_PARAMS)
        
    except Exception as exception:
        LOG.exception(exception)
        raise






















