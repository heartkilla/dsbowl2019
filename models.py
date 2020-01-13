from functools import partial
import warnings

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge

from metric import eval_qwk_lgb_regr, adjust_dist, allocate_to_rate, qwk


warnings.filterwarnings('ignore')


class LGBMModel:
    def __init__(self, params,
                 num_boost_round,
                 early_stopping_rounds,
                 verbose_eval, folds,
                 cols_to_drop,
                 cat_feats,
                 group_col=None,
                 n_rand_val=10):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.folds = folds
        self.group_col = group_col
        self.cols_to_drop = cols_to_drop
        self.cat_feats = cat_feats
        self.n_rand_val = n_rand_val
        self.models = []
        self.tr_means = []
        self.tr_stds = []
        self.scores = {'training': [], 'valid_1': []}
        self.rand_scores = []

    def fit(self, X, y):
        self.columns = X.columns.drop(self.cols_to_drop)
        self.oof_train = np.zeros(len(X))
        self.entire_mean = y.mean()
        self.entire_std = y.std()

        for n_fold, (tr_index, val_index) in enumerate(self.folds.split(X, y, X[self.group_col])):
            print(f'Fold {n_fold + 1}:')

            X_tr, X_val = X.iloc[tr_index], X.iloc[val_index]
            y_tr, y_val = y.iloc[tr_index], y.iloc[val_index]

            #d_tr = lgb.Dataset(X_tr.drop(columns=self.cols_to_drop), y_tr)
            #d_val = lgb.Dataset(X_val.drop(columns=self.cols_to_drop), y_val)

            tr_mean = y_tr.mean()
            self.tr_means.append(tr_mean)
            tr_std = y_tr.std()
            self.tr_stds.append(tr_std)

            model = Ridge(alpha=1).fit(X_tr.drop(columns=self.cols_to_drop), y_tr)

            self.models.append(model)

            val_pred = model.predict(X_val.drop(columns=self.cols_to_drop))
            val_pred = adjust_dist(val_pred, tr_mean, tr_std)
            self.oof_train[val_index] = val_pred
            val_pred = allocate_to_rate(val_pred)

            for i in range(self.n_rand_val):
                X_rand = X_val.groupby('installation_id').apply(lambda x: x.sample(1, random_state=i)).reset_index(drop=True)
                y_rand = X_rand['accuracy_group']
                rand_pred = model.predict(X_rand.drop(columns=self.cols_to_drop))
                rand_pred = adjust_dist(rand_pred, tr_mean, tr_std)
                rand_pred = allocate_to_rate(rand_pred)
                score = qwk(y_rand, rand_pred)
                print(f'Rand QWK score {i + 1}: {score:.6f}')
                self.rand_scores.append(score)

           # for dataset in ['training', 'valid_1']:
            #    self.scores[dataset].append(model.best_score[dataset]['kappa'])

            print('-' * 50)

        print(f'Train mean QWK: {np.mean(self.scores["training"]):.6f}+/-{np.std(self.scores["training"]):.6f}')
        print(f'CV mean QWK: {np.mean(self.scores["valid_1"]):.6f}+/-{np.std(self.scores["valid_1"]):.6f}')
        print(f'CV random QWK: {np.mean(self.rand_scores):.6f}+/-{np.std(self.rand_scores):.6f}')

    def predict(self, X, entire_train_stats=False, raw_values=False):
        X = X[self.columns]
        preds = np.zeros(X.shape[0])

        for i, model in enumerate(self.models):
            pred = model.predict(X)
            if not entire_train_stats:
                pred = adjust_dist(pred, self.tr_means[i], self.tr_stds[i])
            preds += pred
        preds /= len(self.models)

        if entire_train_stats:
            preds = adjust_dist(preds, self.entire_mean, self.entire_std)

        if not raw_values:
            preds = allocate_to_rate(preds)

        return preds
