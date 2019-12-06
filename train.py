import pickle

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

import config
from models import LGBMModel


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_train_path)

    useless_events = [2081, 2070, 2075, 4050, 4080]
    useless_event_feats = []
    for col in X.columns:
        for ev in useless_events:
            if str(ev) in col:
                useless_event_feats.append(col)

    count_cols = [col for col in X.columns if 'count' in col]
    for col in count_cols:
        X[col] = X[col] / X['accumulated_actions']

    to_drop = ['accumulated_accuracy_group', 'accumulated_correct_attempts', 'accumulated_uncorrect_attempts']
    cols_to_drop = [col for col in X.columns if ('event_id' in col or col in to_drop)]
    X = X.drop(columns=cols_to_drop + useless_event_feats)

    print(X.shape)

    y = X['accuracy_group']

    model = LGBMModel(params=config.lgb_params,
                      folds=GroupKFold(n_splits=config.n_folds),  # shuffle=True, random_state=73),
                      cols_to_drop=['installation_id', 'accuracy_group', 'accuracy'],
                      group_col='installation_id',
                      **config.lgb_train_params)

    model.fit(X, y)

    with open('./checkpoints/model.pkl', 'wb') as fout:
            pickle.dump(model, fout)
