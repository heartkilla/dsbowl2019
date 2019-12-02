import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
from models import LGBMModel


if __name__ == '__main__':
    X_train = pd.read_csv(config.preprocessed_train_path)
    X_test = pd.read_csv(config.preprocessed_test_path)

    X = pd.concat([X_train, X_test])
    y = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])

    model = LGBMModel(params=config.lgb_params,
                      folds=StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=73),
                      cols_to_drop=['installation_id', 'accuracy_group'],
                      group_col=None,
                      **config.lgb_train_params)

    model.fit(X, y)
