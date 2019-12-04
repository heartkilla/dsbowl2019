from functools import partial
import warnings

import numpy as np
import lightgbm as lgb

from metric import eval_qwk_lgb_regr, allocate_to_rate, qwk


warnings.filterwarnings("ignore")


class LGBMModel:
    def __init__(self, params, num_boost_round,
                 early_stopping_rounds,
                 verbose_eval, folds,
                 cols_to_drop, group_col=None):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.folds = folds
        self.group_col = group_col
        self.cols_to_drop = cols_to_drop
        self.models = []
        self.tr_means = []
        self.tr_stds = []
        self.scores = {'training': [], 'valid_1': []}

    def fit(self, X, y):
        self.columns = X.columns.drop(self.cols_to_drop)
        self.oof_train = np.zeros(len(X))

        for n_fold, (tr_index, val_index) in enumerate(self.folds.split(X, y, X[self.group_col])):
            print(f'Fold {n_fold + 1}:')

            X_tr, X_val = X.iloc[tr_index], X.iloc[val_index]
            y_tr, y_val = y.iloc[tr_index], y.iloc[val_index]

            X_tr, X_val = X_tr.drop(columns=self.cols_to_drop), X_val.drop(columns=self.cols_to_drop)

            d_tr = lgb.Dataset(X_tr, y_tr)
            d_val = lgb.Dataset(X_val, y_val)

            tr_mean = y_tr.mean()
            self.tr_means.append(tr_mean)
            tr_std = y_tr.std()
            self.tr_stds.append(tr_std)

            model = lgb.train(self.params, d_tr,
                              valid_sets=[d_tr, d_val],
                              feval=partial(eval_qwk_lgb_regr, tr_mean=tr_mean, tr_std=tr_std),
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping_rounds,
                              verbose_eval=self.verbose_eval,
                              categorical_feature=['world'])

            self.models.append(model)

            val_pred = model.predict(X_val)
            val_pred = tr_mean + (val_pred - tr_mean) / (val_pred.std() / tr_std)
            thresholds = [0.5, 1.5, 2.5]
            val_pred = allocate_to_rate(val_pred, thresholds)
            self.oof_train[val_index] = val_pred

            for dataset in ['training', 'valid_1']:
                self.scores[dataset].append(model.best_score[dataset]['kappa'])

            print('-' * 50)

        print(f'Train mean QWK: {np.mean(self.scores["training"]):.6f}+/-{np.std(self.scores["training"]):.6f}')
        print(f'CV mean QWK: {np.mean(self.scores["valid_1"]):.6f}+/-{np.std(self.scores["valid_1"]):.6f}')
        print(f'CV OOF QWK: {qwk(y, self.oof_train):.6f}')

    def predict(self, X):
        X = X.drop(columns=self.cols_to_drop)[self.columns]
        preds = np.zeros(X.shape[0])

        for i, model in enumerate(self.models):
            pred = model.predict(X)
            pred = self.tr_means[i] + (pred - self.tr_means[i]) / (pred.std() / self.tr_stds[i])
            preds += pred
        preds /= len(self.models)

        thresholds = [0.5, 1.5, 2.5]
        preds = allocate_to_rate(preds, thresholds)

        return preds
