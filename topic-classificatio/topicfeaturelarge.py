# %%
import os
import copy

import torch
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from transformers import AutoModel, AutoTokenizer


PATH = "/home/mcq2/GitHub/aes2/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pd.read_csv(PATH + "train.csv")


def predict_chunk(train: pd.DataFrame) -> pd.DataFrame:

    model_0 = AutoModelForSequenceClassification.from_pretrained("/home/mcq2/GitHub/aes2/topic-classificatio/output/fold_0/checkpoint-812")
    model_1 = AutoModelForSequenceClassification.from_pretrained("/home/mcq2/GitHub/aes2/topic-classificatio/output/fold_1/checkpoint-812")


    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("/home/mcq2/GitHub/aes2/topic-classificatio/output/fold_0/checkpoint-812")


    diff_ds = Dataset.from_pandas(train)
    diff_ds = diff_ds.map(lambda i: tokenizer(i["full_text"], max_length=1024, truncation=True), batched=True)

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions = eval_pred.predictions
        y_true = eval_pred.label_ids
        y_pred = predictions.argmax(-1)
        acc = accuracy_score(y_true, y_pred)
        return {"acc": acc}

    args = TrainingArguments(
        output_dir="output",
        report_to="none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        num_train_epochs=1,
        weight_decay=0.01,
        optim="adamw_torch",
        fp16=torch.cuda.is_available())
    
    eval_ds = diff_ds
    train_ds = diff_ds

    trainer_0 = Trainer(
        args=args, 
        model=model_0,
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics
    )

    trainer_1 = Trainer(
        args=args, 
        model=model_1,
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics
    )

    test_preds_0 = trainer_0.predict(diff_ds)
    test_preds_1 = trainer_1.predict(diff_ds)
    predictions = []
    predictions.append(test_preds_0.predictions)
    predictions.append(test_preds_1.predictions)

    predictions = np.stack(predictions, axis=0).mean(axis=0)
    predictions = predictions.argmax(-1)
    essay_id = diff_ds["essay_id"]
    # convert prompt id back to prompt name
    result_df = pd.DataFrame({"essay_id": essay_id, "topic":predictions })
    

    return result_df


# %%
if __name__ == "__main__":
    submission_1 = predict_chunk(train)


