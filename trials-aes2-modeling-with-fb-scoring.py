# %%
import aes2_added_fb_prize_as_features_preprocessing

# %%
import gc
# from catboost import CatBoostClassifier, Pool, CatBoostRegressor
gc.collect()

# %%
import pickle

with open("/kaggle/input/train_data/train_feats.pickle", "rb") as f:
    train_feats = pickle.load(f)
with open("/kaggle/input/aes2_train_data/X.pickle", "rb") as f:
    X = pickle.load(f)
with open("/kaggle/input/train_data/y.pickle", "rb") as f:
    y = pickle.load(f)
with open("/kaggle/input/train_data/y_split.pickle", "rb") as f:
    y_split = pickle.load(f)
with open(
    "/kaggle/input/train_data/feature_select.pickle", "rb"
) as f:
    feature_select = pickle.load(f)
    
aes2_added_fb_prize_as_features_preprocessing.feature_select = feature_select

# %%
train_feats.iloc[:5, -7:]

# %%
import numpy as np

X = train_feats[feature_select].astype(np.float32).values

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, cohen_kappa_score
from aes2_added_fb_prize_as_features_preprocessing import *
tuner_params = nni.get_next_parameter()
n_splits = tuner_params['n_splits']
models = []
predictions = []
f1_scores = []
kappa_scores = []
from xgboost import DMatrix

class Predictor:
    def __init__(self, models: list):
        self.models = models
#         self.xgb_boost_best_iter = models[1].
    def predict(self, X):
        n_models = len(self.models)
        predicted = None
        # n = 0.749
        a = 0.427
        b = 0.342
        c = 0.231
        for i, model in enumerate(self.models):
            if i == 0: # LightGBM weight
                predicted = a*model.predict(X)
            elif i == 1: # XGBoost weight
                if not isinstance(X, DMatrix):
                    X = xgb.DMatrix(X)
                predicted += b*model.predict(X)
            else: # Catboost weight
                predicted += c*model.predict(X)
        return predicted

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
xgboost_best_iters = []
light_best_iters = []

for i, (train_index, test_index) in enumerate(skf.split(X, y_split), 1):
    # Split the data into training and testing sets for this fold
    print('fold',i)
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]
    callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=75,first_metric_only=True)]
    light = lgb.LGBMRegressor(
            objective = qwk_obj,
            metrics = 'None',
            learning_rate = tuner_params['learning_rate'],
            max_depth = tuner_params['max_depth'],
            num_leaves = tuner_params['num_leaves'],
            colsample_bytree=tuner_params['colsample_bytree'],
            reg_alpha = tuner_params['reg_alpha'],
            reg_lambda = tuner_params['reg_lambda'],
            n_estimators=tuner_params['n_estimators'],
            random_state=42,
            extra_trees=True,
            class_weight='balanced',
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
    light_best_iters.append(light.best_iteration_)
    xgb_callbacks = [
        xgb.callback.EvaluationMonitor(period=25),
        xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
    ]
    xgb_regressor = xgb.XGBRegressor(
        objective = qwk_obj,
        metrics = 'None',
        learning_rate = tuner_params['learning_rate'],
        max_depth = tuner_params['max_depth'],
        num_leaves = tuner_params['num_leaves'],
        colsample_bytree=tuner_params['colsample_bytree'],
        reg_alpha = tuner_params['reg_alpha'],
        reg_lambda = tuner_params['reg_lambda'],
        n_estimators=tuner_params['n_estimators'],
        random_state=42,
        extra_trees=True,
        class_weight='balanced',
        tree_method="gpu_hist",
        # device="gpu" if CUDA_AVAILABLE else "cpu",
        gpu_id = 7
    #             device='gpu',
    #             verbosity = 1
    )
    
    xgb_callbacks = [
        xgb.callback.EvaluationMonitor(period=100),
        xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
    ]
    xgb_regressor.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
        eval_metric=quadratic_weighted_kappa,
        callbacks=xgb_callbacks
    )
    xgboost_best_iters.append(xgb_regressor.get_booster().best_iteration)
    
    # CAT
    cat_model = CatBoostRegressor(
        iterations= tuner_params['iterations'],
        learning_rate = tuner_params['learning_rate'],
        depth = tuner_params['max_depth'],
        bootstrap_type='Bernoulli',
        subsample=tuner_params['subsample'],
        l2_leaf_reg = tuner_params['l2_leaf_reg'],
        task_type = 'GPU',
        devices= '7:8', # 在Kaggle上记得删
        objective = 'RMSE',
        eval_metric = 'RMSE',
        loss_function="RMSE")
    train_pool = Pool(data = X_train_fold, label = y_train_fold)
    valid_pool = Pool(data = X_test_fold, label = y_test_fold)

    cat_model.fit(train_pool,
                verbose=100,
                eval_set=valid_pool,
                early_stopping_rounds=75
                )
    
    # cat_model.save_model(f'kaggle/out/aes-catboost/fold_{i}.cbm')
    # print('\nFold_{} CatBoost Model saved.\n'.format(i))


    
    
    
    predictor = Predictor([light, xgb_regressor, cat_model])

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
    kappa_scores.append(kappa_fold)
#         predictor.booster_.save_model(f'fold_{i}.txt')
    cm = confusion_matrix(y_test_fold_int, predictions_fold, labels=[x for x in range(1,7)])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[x for x in range(1,7)]
    )
    disp.plot()
    plt.show()
    print(f'F1 score across fold: {f1_fold}')
    print(f'Cohen kappa score across fold: {kappa_fold}')

    gc.collect()
    #if ENABLE_DONT_WASTE_YOUR_RUN_TIME:
    #    break


# %%


# %%
mean_f1_score = np.mean(f1_scores)
mean_kappa_score = np.mean(kappa_scores)
# Print the mean scores
print(f'Mean F1 score across {n_splits} folds: {mean_f1_score}')
print(f'Mean Cohen kappa score across {n_splits} folds: {mean_kappa_score}')
print(f"XGBoost mean best iters: {sum(xgboost_best_iters)/len(xgboost_best_iters)}")
print(f"LightBoost mean best iters: {sum(light_best_iters)/len(light_best_iters)}")


