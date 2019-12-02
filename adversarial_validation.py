import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

import config


if __name__ == '__main__':
    X_train = pd.read_csv(config.preprocessed_train_path)
    X_test = pd.read_csv(config.preprocessed_test_path)

    cols_to_drop = ['installation_id', 'accuracy_group']

    X = pd.concat([X_train, X_test])
    y = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for i, (tr_index, val_index) in enumerate(folds.split(X, y, X['installation_id'])):
        X_tr, X_val = X.iloc[tr_index].drop(columns=cols_to_drop), X.iloc[val_index].drop(columns=cols_to_drop)
        y_tr, y_val = y[tr_index], y[val_index]

        model = LGBMClassifier().fit(X_tr, y_tr)

        print(f'Fold {i + 1} ROC AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])}')

        print(X_tr.columns[np.argsort(model.feature_importances_)[:10:-1]])
