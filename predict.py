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
    X = X.drop(columns=['hour'])

    X['weekday_sin'] = X['weekday'].map(lambda x: np.sin(2 * np.pi * x / 6))
    X['weekday_cos'] = X['weekday'].map(lambda x: np.cos(2 * np.pi * x / 6))
    X = X.drop(columns=['weekday'])

    X['day_sin'] = X['day'].map(lambda x: np.sin(2 * np.pi * x / 6))
    X['day_cos'] = X['day'].map(lambda x: np.cos(2 * np.pi * x / 6))
    X = X.drop(columns=['day'])

    X['mean_time_per_day'] = X['total_time'] / X['days_since_installation']
    X['current_title_mean_time'] = X['current_title_total_time'] / X['current_title_count']
    X['current_world_mean_time'] = X['current_world_total_time'] / X['current_world_count']
    X['last_mean_accuracy'] = X[['last_accuracy_Bird Measurer (Assessment)',
                                 'last_accuracy_Cart Balancer (Assessment)',
                                 'last_accuracy_Cauldron Filler (Assessment)',
                                 'last_accuracy_Chest Sorter (Assessment)',
                                 'last_accuracy_Mushroom Sorter (Assessment)']].mean(axis=1)
    X['ratio_of_life_in_game'] = X['total_time'] / X['sec_since_installation']
    X['time_per_session'] = X['total_time'] / X['accumulated_sessions']
    X['sessions_per_day'] = X['accumulated_sessions'] / X['days_since_installation']
    X['events_per_session'] = X['accumulated_actions'] / X['accumulated_sessions']
    X['time_per_event'] = X['total_time'] / X['accumulated_actions']
    X['events_per_day'] = X['accumulated_actions'] / X['days_since_installation']
    preds = model.predict(X)

    sample_submission['accuracy_group'] = preds.astype(int)
    sample_submission.to_csv('./data/submission.csv', index=False)

    print(sample_submission['accuracy_group'].value_counts(normalize=True))
