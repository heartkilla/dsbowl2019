import pickle

import numpy as np
import pandas as pd

import config


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_test_path)
    sample_submission = pd.read_csv(config.sample_sub_path)

    with open('./checkpoints/model.pkl', 'rb') as fin:
        model = pickle.load(fin)

    X['hour_sin'] = X['hour'].map(lambda x: np.sin(2 * np.pi * x / 23))
    X['hour_cos'] = X['hour'].map(lambda x: np.cos(2 * np.pi * x / 23))

    X['weekday_sin'] = X['weekday'].map(lambda x: np.sin(2 * np.pi * x / 6))
    X['weekday_cos'] = X['weekday'].map(lambda x: np.cos(2 * np.pi * x / 6))

    X['mean_time_per_day'] = X['total_time'] / X['days_since_installation']

    preds = model.predict(X)

    sample_submission['accuracy_group'] = preds.astype(int)
    sample_submission.to_csv('./data/submission.csv', index=False)

    print(sample_submission['accuracy_group'].value_counts(normalize=True))
