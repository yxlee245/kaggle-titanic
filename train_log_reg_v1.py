import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import mlflow
from util import (
    DataPrepUtil, initialize_column_transformer, initialize_pipeline, plot_roc_curve,
    plot_learning_curve, config, record_hyperparameters
)


if __name__ == "__main__":
    data_prep_util = DataPrepUtil()
    X_train, y_train = data_prep_util.load_train_data(config.get('paths').get('train_data'))
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=config.get('test_size'), random_state=config.get('random_state'),
        stratify=y_train
    )
    # X_test, _ = data_prep_util.transform_data(config.get('paths').get('test_data'), store_ids=True)
    
    mlflow.set_experiment(config.get('mlflow').get('exp_name'))
    RUN_NAME = os.path.basename(__file__).split('.')[0]
    with mlflow.start_run(run_name=RUN_NAME):
        print('Starting run {}...'.format(RUN_NAME))
        mlflow.log_param('test_size', config.get('test_size'))
        pipeline = initialize_pipeline(
            initialize_column_transformer(scale_values=True),
            LogisticRegression(
                solver=config.get('hyperparams').get('log_reg').get('solver'),
                random_state=config.get('random_state'),
                max_iter=config.get('hyperparams').get('log_reg').get('max_iter'),
                tol=config.get('hyperparams').get('log_reg').get('tol')
            )
        )
        param_grid = {
            'model__penalty': config.get('hyperparams').get('log_reg').get('penalty'),
            'model__C': config.get('hyperparams').get('log_reg').get('C')
        }
        gs_clf = GridSearchCV(
            pipeline, param_grid=param_grid,
            scoring=config.get('scoring'), cv=config.get('k'), verbose=1
        )
        gs_clf.fit(X_train, y_train)
        mlflow.log_params(gs_clf.best_params_)
        mlflow.log_metrics({
                '_'.join([config.get('scoring'), 'train']): gs_clf.score(X_train, y_train),
                '_'.join([config.get('scoring'), 'test']): gs_clf.score(X_test, y_test)
        })

        print('Plotting ROC curve...')
        img_fn = 'roc_{}.png'.format(RUN_NAME)
        plot_roc_curve(gs_clf, X_test, y_test, img_fn)
        
        # print('Plotting learning curve...')
        # img_fn = 'learning_curve_{}.png'.format(RUN_NAME)
        # plot_learning_curve(gs_clf, X_train, y_train, img_fn)

        print('Recording hyperparameters used...')
        txt_fn = 'hyperparams_{}.txt'.format(RUN_NAME)
        record_hyperparameters(txt_fn, config.get('hyperparams').get('log_reg'))