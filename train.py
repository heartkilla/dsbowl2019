import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

import config
from models import LGBMModel


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_train_path)
    X_add = pd.read_csv(config.preprocessed_test_for_train_path)
    X = pd.concat([X, X_add])

    #useless_events = [2081, 2070, 2075, 4050, 4080, 4110, 4220, 2050, 4235, 2035, 5010, 4230, 5000, 2040,
    #                  4095, 2025, 2083, 2060]
    useless_event_feats = []
    #for col in X.columns:
    #    for ev in useless_events:
    #        if str(ev) in col:
    #            useless_event_feats.append(col)

    X['hour_sin'] = X['hour'].map(lambda x: np.sin(2 * np.pi * x / 23))
    X['hour_cos'] = X['hour'].map(lambda x: np.cos(2 * np.pi * x / 23))
    X = X.drop(columns=['hour'])

    X['weekday_sin'] = X['weekday'].map(lambda x: np.sin(2 * np.pi * x / 6))
    X['weekday_cos'] = X['weekday'].map(lambda x: np.cos(2 * np.pi * x / 6))
    X = X.drop(columns=['weekday'])

    X['day_sin'] = X['day'].map(lambda x: np.sin(2 * np.pi * x / 6))
    X['day_cos'] = X['day'].map(lambda x: np.cos(2 * np.pi * x / 6))
    X = X.drop(columns=['day'])

    X['mean_time_per_day'] = X['total_time_sum'] / X['days_since_installation']
    X['current_title_mean_time'] = X['current_title_total_time'] / X['current_title_count']
    X['current_world_mean_time'] = X['current_world_total_time'] / X['current_world_count']
    X['last_mean_accuracy'] = X[['last_accuracy_Bird Measurer (Assessment)',
                                 'last_accuracy_Cart Balancer (Assessment)',
                                 'last_accuracy_Cauldron Filler (Assessment)',
                                 'last_accuracy_Chest Sorter (Assessment)',
                                 'last_accuracy_Mushroom Sorter (Assessment)']].mean(axis=1)
    X['ratio_of_life_in_game'] = X['total_time_sum'] / X['sec_since_installation']
    X['time_per_session'] = X['total_time_sum'] / X['accumulated_sessions']
    X['sessions_per_day'] = X['accumulated_sessions'] / X['days_since_installation']
    X['events_per_session'] = X['accumulated_actions'] / X['accumulated_sessions']
    X['time_per_event'] = X['total_time_sum'] / X['accumulated_actions']
    X['events_per_day'] = X['accumulated_actions'] / X['days_since_installation']

    cols_to_drop = [col for col in X.columns if ('event_id' in col)]

    count_cols = [col for col in X.columns if 'count' in col]
    for col in count_cols:
        #if 'event_code' in col:
        X[col] = X[col] / X['accumulated_actions']
        #else:
        #    X[col] = X[col] / X['accumulated_sessions']

    X = X.drop(columns=cols_to_drop + useless_event_feats).replace({np.inf: 0})

    print(X.shape)

    y = X['accuracy_group']

    model = LGBMModel(params=config.lgb_params,
                      folds=GroupKFold(n_splits=config.n_folds), #shuffle=True, random_state=73),
                      cols_to_drop=['installation_id', 'accuracy_group'],
                      group_col='installation_id',
                      **config.lgb_train_params)

    model.fit(X, y)

    with open('./checkpoints/model.pkl', 'wb') as fout:
            pickle.dump(model, fout)
