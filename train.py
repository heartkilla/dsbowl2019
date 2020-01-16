import pickle

import pandas as pd
from sklearn.model_selection import GroupKFold

import config
import feature_names
from models import LGBMModel, RidgeStacker
from metric import allocate_to_rate
from preprocessing import add_final_features


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_train_path)
    X_add = pd.read_csv(config.preprocessed_test_for_train_path)
    X = pd.concat([X, X_add])

    X = add_final_features(X)

    y = X['accuracy_group']

    model_1 = LGBMModel(params=config.lgb_params_1,
                        folds=GroupKFold(n_splits=config.n_folds),
                        cols_to_drop=['installation_id', 'accuracy_group'],
                        group_col='installation_id',
                        **config.lgb_train_params_1)
    model_1.fit(X, y)

    config.lgb_params_1['feature_fraction_seed'] = 305
    config.lgb_params_1['seed'] = 305
    model_1_new = LGBMModel(params=config.lgb_params_1,
                            folds=GroupKFold(n_splits=config.n_folds),
                            cols_to_drop=['installation_id', 'accuracy_group'],
                            group_col='installation_id',
                            **config.lgb_train_params_1)
    model_1_new.fit(X, y)

    X_2 = X[['accuracy_group', 'installation_id'] + feature_names.model_2_feature_names]
    model_2 = LGBMModel(params=config.lgb_params_2,
                        folds=GroupKFold(n_splits=config.n_folds),
                        cols_to_drop=['installation_id', 'accuracy_group'],
                        group_col='installation_id',
                        **config.lgb_train_params_2)
    model_2.fit(X_2, y)

    X_3 = X[['accuracy_group', 'installation_id'] + feature_names.model_3_feature_names]
    model_3 = LGBMModel(params=config.lgb_params_3,
                        folds=GroupKFold(n_splits=config.n_folds),
                        cols_to_drop=['installation_id', 'accuracy_group'],
                        group_col='installation_id',
                        **config.lgb_train_params_3)
    model_3.fit(X_3, y)

    config.lgb_params_3['feature_fraction_seed'] = 123
    config.lgb_params_3['seed'] = 123
    model_3_new = LGBMModel(params=config.lgb_params_3,
                            folds=GroupKFold(n_splits=config.n_folds),
                            cols_to_drop=['installation_id', 'accuracy_group'],
                            group_col='installation_id',
                            **config.lgb_train_params_3)
    model_3_new.fit(X_3, y)

    X_stack = X[['accuracy_group', 'installation_id']]
    X_stack['feat_1'] = allocate_to_rate(model_1.oof_train)
    X_stack['feat_2'] = allocate_to_rate(model_2.oof_train)
    X_stack['feat_3'] = allocate_to_rate(model_3.oof_train)
    X_stack['feat_1_new'] = allocate_to_rate(model_1_new.oof_train)
    X_stack['feat_3_new'] = allocate_to_rate(model_3_new.oof_train)

    stacker = RidgeStacker(alpha=config.ridge_alpha,
                           folds=GroupKFold(n_splits=config.n_folds),
                           cols_to_drop=['installation_id', 'accuracy_group'],
                           group_col='installation_id')
    stacker.fit(X_stack, y)

    with open('./checkpoints/model_1.pkl', 'wb') as fout:
            pickle.dump(model_1, fout)
    with open('./checkpoints/model_2.pkl', 'wb') as fout:
            pickle.dump(model_2, fout)
    with open('./checkpoints/model_3.pkl', 'wb') as fout:
            pickle.dump(model_3, fout)
    with open('./checkpoints/model_1_new.pkl', 'wb') as fout:
            pickle.dump(model_1_new, fout)
    with open('./checkpoints/model_3_new.pkl', 'wb') as fout:
            pickle.dump(model_3_new, fout)
    with open('./checkpoints/stacker.pkl', 'wb') as fout:
            pickle.dump(stacker, fout)
