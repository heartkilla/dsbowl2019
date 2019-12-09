from functools import partial
import warnings

import numpy as np
import lightgbm as lgb

from metric import eval_qwk_lgb_regr, allocate_to_rate, qwk

import random
from collections import Counter, defaultdict


warnings.filterwarnings("ignore")

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


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
        self.map_groups = ['current_world_mean_time',
                           'current_title_mean_time',
                           'last_mean_accuracy',
                           'current_title_count',
                           'accumulated_accuracy',
                           'current_world_game_time',
                           'total_time',
                           'accumulated_accuracy_group',
                           'mean_time_per_day']
        self.title_mappings = []
        self.world_mappings = []

    def fit(self, X, y):
        self.columns = X.columns.drop(self.cols_to_drop)
        self.oof_train = np.zeros(len(X))

        oof_rand = []
        oof_rand_true = []
        rand_scores = []

        for n_fold, (tr_index, val_index) in enumerate(stratified_group_k_fold(X, y, X[self.group_col], k=5)):
            print(f'Fold {n_fold + 1}:')

            X_tr, X_val = X.iloc[tr_index], X.iloc[val_index]
            y_tr, y_val = y.iloc[tr_index], y.iloc[val_index]

            X_rands = [X_val.groupby('installation_id').apply(lambda x: x.sample(1, random_state=i)).reset_index(drop=True) for i in range(5)]
            y_rands = [X_rand['accuracy_group'] for X_rand in X_rands]
            X_rands = [X_rand.drop(columns=self.cols_to_drop) for X_rand in X_rands]

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
                              verbose_eval=self.verbose_eval)
                              #categorical_feature=['world', 'title'])

            self.models.append(model)

            val_pred = model.predict(X_val)
            val_pred = tr_mean + (val_pred - val_pred.mean()) / (val_pred.std() / tr_std)
            thresholds = [0.5, 1.5, 2.5]
            val_pred = allocate_to_rate(val_pred, thresholds)
            self.oof_train[val_index] = val_pred

            for i in range(5):
                val_pred = model.predict(X_rands[i])
                val_pred = tr_mean + (val_pred - val_pred.mean()) / (val_pred.std() / tr_std)
                thresholds = [0.5, 1.5, 2.5]
                val_pred = allocate_to_rate(val_pred, thresholds)
                oof_rand.extend(val_pred)
                oof_rand_true.extend(y_rands[i])
                score = qwk(y_rands[i], val_pred)
                print(f'rand qwk: {score:.6f}')
                rand_scores.append(score)

            for dataset in ['training', 'valid_1']:
                self.scores[dataset].append(model.best_score[dataset]['kappa'])

            print('-' * 50)

        print(f'Train mean QWK: {np.mean(self.scores["training"]):.6f}+/-{np.std(self.scores["training"]):.6f}')
        print(f'CV mean QWK: {np.mean(self.scores["valid_1"]):.6f}+/-{np.std(self.scores["valid_1"]):.6f}')
        print(f'CV OOF QWK: {qwk(y, self.oof_train):.6f}')

        print(f'CV random QWK: {np.mean(rand_scores):.6f}+/-{np.std(rand_scores):.6f}')
        print(f'CV OOF random QWK: {qwk(oof_rand_true, oof_rand):.6f}')

    def predict(self, X):
        X = X.drop(columns=self.cols_to_drop)[self.columns]
        preds = np.zeros(X.shape[0])

        for i, model in enumerate(self.models):
            pred = model.predict(X)
            pred = self.tr_means[i] + (pred - pred.mean()) / (pred.std() / self.tr_stds[i])
            preds += pred
        preds /= len(self.models)

        thresholds = [0.5, 1.5, 2.5]
        preds = allocate_to_rate(preds, thresholds)

        return preds
