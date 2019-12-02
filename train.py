import pickle

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

import config
from models import LGBMModel


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_train_path)

    to_drop = ['accumulated_actions', 'accumulated_accuracy_group', 'accumulated_correct_attempts', 'accumulated_uncorrect_attempts']
    cols_to_drop = [col for col in X.columns if ('event_id' in col or '_group_count' in col or col in to_drop)]
    X = X.drop(columns=cols_to_drop)

    y = X['accuracy_group']

    model = LGBMModel(params=config.lgb_params,
                      folds=StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=73),
                      cols_to_drop=['installation_id', 'accuracy_group'],
                      group_col='installation_id',
                      **config.lgb_train_params)

    model.fit(X, y)

    with open('./checkpoints/model.pkl', 'wb') as fout:
            pickle.dump(model, fout)
