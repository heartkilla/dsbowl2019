import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

import config
from models import LGBMModel


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_train_path)

    X['hour_sin'] = X['hour'].map(lambda x: np.sin(2 * np.pi * x / 23))
    X['hour_cos'] = X['hour'].map(lambda x: np.cos(2 * np.pi * x / 23))

    X['weekday_sin'] = X['weekday'].map(lambda x: np.sin(2 * np.pi * x / 6))
    X['weekday_cos'] = X['weekday'].map(lambda x: np.cos(2 * np.pi * x / 6))

    X['mean_time_per_day'] = X['total_time'] / X['days_since_installation']

    to_drop = ['accumulated_actions', 'day'] + [col for col in X.columns if ('changed' in col or ('title_' in col and 'current' not in col))]

    X = X.drop(columns=to_drop)

    print(X.shape)

    y = X['accuracy_group']

    model = LGBMModel(params=config.lgb_params,
                      folds=GroupKFold(n_splits=config.n_folds),  # shuffle=True, random_state=73),
                      cols_to_drop=['installation_id', 'accuracy_group'],
                      group_col='installation_id',
                      **config.lgb_train_params)

    model.fit(X, y)

    with open('./checkpoints/model.pkl', 'wb') as fout:
            pickle.dump(model, fout)
