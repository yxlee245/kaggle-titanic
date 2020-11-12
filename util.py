import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
import mlflow

with open('config.json', 'r') as json_file:
    config = json.load(json_file)


class DataPrepUtil:
    imputer_mode = SimpleImputer(strategy='most_frequent')
    imputer_mean = SimpleImputer(strategy='mean')
    train_data_loaded = False

    def load_train_data(self, train_data_path: str):
        train_data = pd.read_csv(train_data_path)
        train_data = train_data.drop(columns=config.get('columns').get('to_drop'))
        train_data[config.get('columns').get('discrete')] = (
            self.imputer_mode.fit_transform(train_data[config.get('columns').get('discrete')])
        )
        train_data[config.get('columns').get('continuous')] = (
            self.imputer_mean.fit_transform(train_data[config.get('columns').get('continuous')])
        )
        y_train = train_data[config.get('columns').get('label')]
        X_train = train_data.drop(columns=[config.get('columns').get('label')])
        self.train_data_loaded = True
        return X_train, y_train

    def transform_data(self, data_path: str, store_ids: bool = False):
        if not self.train_data_loaded:
            raise Exception('Training data has not been loaded yet, so unable to transform data')
        self.passengerids = None
        data = pd.read_csv(data_path)
        if store_ids:
            self.passengerids = data[config.get('columns').get('id')]
        data = data.drop(columns=config.get('columns').get('to_drop'))
        data[config.get('columns').get('discrete')] = (
            self.imputer_mode.transform(data[config.get('columns').get('discrete')])
        )
        data[config.get('columns').get('continuous')] = (
            self.imputer_mean.transform(data[config.get('columns').get('continuous')])
        )
        if config.get('columns').get('label') in data:
            y = data[config.get('columns').get('label')]
            X = data.drop(columns=[config.get('columns').get('label')])
            return X, y
        return data, None

    def get_ids(self):
        if not isinstance(self.passengerids, pd.Series):
            print('No passenger ids stored')
        return self.passengerids


def initialize_column_transformer(scale_values=False):
    transformers = [
        ('ohe', OneHotEncoder(sparse=False, drop='first'), config.get('columns').get('categorical'))
    ]
    if scale_values:
        transformers.append(
            ('standard_scaler', StandardScaler(), config.get('columns').get('continuous'))
        )
    return ColumnTransformer(transformers=transformers, remainder='passthrough')


def initialize_pipeline(column_transformer: ColumnTransformer, model):
    steps = [
        ('col_trans', column_transformer),
        ('model', model)
    ]
    return Pipeline(steps=steps)


def plot_roc_curve(estimator, X, y, img_fn):
    pred_probs = estimator.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_probs)
    auc_score = roc_auc_score(y, pred_probs)
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC score: {:.2f}'.format(auc_score))
    plt.legend(loc='best')
    plt.savefig(img_fn)
    mlflow.log_artifact(img_fn)
    plt.close()
    os.remove(img_fn)

def plot_learning_curve(estimator, X, y, img_fn):
    train_sizes, train_scores, test_scores = (
        learning_curve(estimator, X, y, cv=config.get('k'), scoring=config.get('scoring'))
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(9, 6))
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color='orange'
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color='green'
    )
    plt.plot(train_sizes, train_scores_mean, 'o-', color='orange', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='CV score')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(img_fn)
    mlflow.log_artifact(img_fn)
    plt.close()
    os.remove(img_fn)


def record_hyperparameters(txt_fn: str, hyperparams: dict):
    with open(txt_fn, 'w') as txt_file:
        for key, value in hyperparams.items():
            txt_file.write('{}: {}\n'.format(key, value))
    mlflow.log_artifact(txt_fn)
    os.remove(txt_fn)