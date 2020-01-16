import copy
import pickle
from collections import Counter
from shutil import copyfile
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
from scipy.stats import skew, mode

import config


class CategoricalEncoder:
    def __init__(self, cat_cols, encoder=OrdinalEncoder(dtype=np.int32)):
        self.cat_cols = cat_cols
        self.encoder = encoder

    def fit(self, data):
        self.encoder.fit(data[self.cat_cols])

    def transform(self, data):
        data[self.cat_cols] = self.encoder.transform(data[self.cat_cols])
        return data


class CustomCounter:
    def __init__(self, count_cols):
        self.count_cols = count_cols

    def fit(self, data):
        self.count_unique_keys = {count_col: data[count_col].unique() for count_col in self.count_cols}

    def reset(self):
        self.counters = {count_col: Counter({count_col + '_' + str(key) + '_count': 0 for key in self.count_unique_keys[count_col]}) for count_col in self.count_unique_keys}


def get_interaction(data, attr_1, attr_2):
    data[attr_1 + '_' + attr_2] = data[attr_1].astype(str) + '_' + data[attr_2].astype(str)
    return data


def preprocess_inst(ins_group, custom_counter, dataset):
    all_assessments = []

    custom_counter.reset()

    types = ['Clip', 'Activity', 'Assessment', 'Game']
    type_counter = {'type_' + col + '_count': 0 for col in types}
    last_type = 0

    accumulated_correct_attempts = []
    accumulated_uncorrect_attempts = []
    accumulated_accuracy = 0
    assessments = ['Cart Balancer (Assessment)',
                   'Cauldron Filler (Assessment)',
                   'Chest Sorter (Assessment)',
                   'Mushroom Sorter (Assessment)',
                   'Bird Measurer (Assessment)']
    last_accuracy_title = {'last_accuracy_' + title: np.nan for title in assessments}
    accuracy_group_count = {f'{i}_group_count': 0 for i in range(4)}
    accumulated_accuracy_group = 0
    accumulated_actions = 0
    accumulated_sessions = 0

    durations = []
    total_times = []
    title_times = {title + '_title_total_time': 0 for title in custom_counter.count_unique_keys['title']}

    worlds = ['NONE', 'MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY']
    world_times = {world + '_world_total_time': 0 for world in worlds}
    world_game_times = {world + '_world_game_time': 0 for world in worlds}
    world_activity_times = {world + '_world_activity_time': 0 for world in worlds}

    accumulated_misses = 0
    all_types_accumulated_correct_attempts = 0
    all_types_accumulated_uncorrect_attempts = 0
    change_type_count = 0

    accuracies = []
    accuracy_groups = []
    last_accuracy_group_title = {'last_accuracy_group_' + title: np.nan for title in assessments}

    session_activity_lens = []

    weekdays = {'weekday_' + str(i): 0 for i in range(7)}
    hours = {'hour_' + str(i): 0 for i in range(24)}

    assess_titles = []
    titles = []

    k = 0
    for game_session, session_group in ins_group.groupby('game_session',
                                                         sort=False):
        session_inst_id = session_group['installation_id'].iloc[0]
        session_type = session_group['type'].iloc[0]
        session_title = session_group['title'].iloc[0]
        session_world = session_group['world'].iloc[0]
        duration = (session_group.iloc[-1, 2] - session_group.iloc[0, 2]).seconds if len(session_group) > 1 else 0

        if k == 0:
            install_time = session_group['timestamp'].iloc[0]

        if session_title == 'Bird Measurer (Assessment)':
            attempt_code = 4110
        else:
            attempt_code = 4100

        if session_type == 'Assessment' and (dataset != 'train' or len(session_group) > 1):
            features = {}
            features['installation_id'] = session_inst_id
            features['title'] = session_title
            features['world'] = session_world
            for counter in custom_counter.counters.values():
                features.update(counter)
            features.update(type_counter)

            features['accumulated_correct_attempts_mean'] = np.mean(accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_median'] = np.median(accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_sum'] = np.sum(accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_max'] = np.max(accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_min'] = np.min(accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_std'] = np.std(accumulated_correct_attempts) if len(accumulated_correct_attempts) > 1 else 0
            features['accumulated_correct_attempts_skew'] = skew(accumulated_correct_attempts) if len(accumulated_correct_attempts) > 1 else 0
            features['accumulated_correct_attempts_lag'] = accumulated_correct_attempts[-1] if len(accumulated_correct_attempts) > 0 else 0
            features['accumulated_correct_attempts_lag_diff'] = accumulated_correct_attempts[-1] - accumulated_correct_attempts[-2] if len(accumulated_correct_attempts) > 1 else 0

            features['accumulated_uncorrect_attempts_mean'] = np.mean(accumulated_uncorrect_attempts) if accumulated_uncorrect_attempts else 0
            features['accumulated_uncorrect_attempts_median'] = np.median(accumulated_uncorrect_attempts) if accumulated_uncorrect_attempts else 0
            features['accumulated_uncorrect_attempts_sum'] = np.sum(accumulated_uncorrect_attempts) if accumulated_uncorrect_attempts else 0
            features['accumulated_uncorrect_attempts_max'] = np.max(accumulated_uncorrect_attempts) if accumulated_uncorrect_attempts else 0
            features['accumulated_uncorrect_attempts_min'] = np.min(accumulated_uncorrect_attempts) if accumulated_uncorrect_attempts else 0
            features['accumulated_uncorrect_attempts_std'] = np.std(accumulated_uncorrect_attempts) if len(accumulated_uncorrect_attempts) > 1 else 0
            features['accumulated_uncorrect_attempts_skew'] = skew(accumulated_uncorrect_attempts) if len(accumulated_uncorrect_attempts) > 1 else 0
            features['accumulated_uncorrect_attempts_lag'] = accumulated_uncorrect_attempts[-1] if len(accumulated_uncorrect_attempts) > 0 else 0
            features['accumulated_uncorrect_attempts_lag_diff'] = accumulated_uncorrect_attempts[-1] - accumulated_uncorrect_attempts[-2] if len(accumulated_uncorrect_attempts) > 1 else 0

            features['accumulated_accuracy'] = accumulated_accuracy / k if k > 0 else 0
            features.update(last_accuracy_title)
            features.update(last_accuracy_group_title)

            features['assess_duration_mean'] = np.mean(durations) if durations else 0
            features['assess_duration_median'] = np.median(durations) if durations else 0
            features['assess_duration_max'] = np.max(durations) if durations else 0
            features['assess_duration_min'] = np.min(durations) if durations else 0
            features['assess_duration_sum'] = np.sum(durations) if durations else 0
            features['assess_duration_std'] = np.std(durations) if len(durations) > 1 else 0
            features['assess_duration_skew'] = skew(durations) if len(durations) > 1 else 0
            features['assess_duration_lag'] = durations[-1] if len(durations) > 0 else 0
            features['assess_duration_lag_diff'] = durations[-1] - durations[-2] if len(durations) > 1 else 0

            features['accuracies_mean'] = np.mean(accuracies) if accuracies else np.nan
            features['accuracies_median'] = np.median(accuracies) if accuracies else np.nan
            features['accuracies_max'] = np.max(accuracies) if accuracies else np.nan
            features['accuracies_min'] = np.min(accuracies) if accuracies else np.nan
            features['accuracies_sum'] = np.sum(accuracies) if accuracies else np.nan
            features['accuracies_std'] = np.std(accuracies) if len(accuracies) > 1 else np.nan
            features['accuracies_skew'] = skew(accuracies) if len(accuracies) > 1 else np.nan
            features['accuracies_lag'] = accuracies[-1] if len(accuracies) > 0 else np.nan
            features['accuracies_lag_diff'] = accuracies[-1] - accuracies[-2] if len(accuracies) > 1 else np.nan

            features['accuracy_groups_mean'] = np.mean(accuracy_groups) if accuracy_groups else np.nan
            features['accuracy_groups_median'] = np.median(accuracy_groups) if accuracy_groups else np.nan
            features['accuracy_groups_max'] = np.max(accuracy_groups) if accuracy_groups else np.nan
            features['accuracy_groups_min'] = np.min(accuracy_groups) if accuracy_groups else np.nan
            features['accuracy_groups_sum'] = np.sum(accuracy_groups) if accuracy_groups else np.nan
            features['accuracy_groups_std'] = np.std(accuracy_groups) if len(accuracy_groups) > 1 else np.nan
            features['accuracy_groups_skew'] = skew(accuracy_groups) if len(accuracy_groups) > 1 else np.nan
            features['accuracy_groups_lag'] = accuracy_groups[-1] if len(accuracy_groups) > 0 else np.nan
            features['accuracy_groups_lag_1'] = accuracy_groups[-2] if len(accuracy_groups) > 1 else np.nan
            features['accuracy_groups_lag_diff'] = accuracy_groups[-1] - accuracy_groups[-2] if len(accuracy_groups) > 1 else np.nan

            features.update(accuracy_group_count)
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / k if k > 0 else 0
            features['accumulated_actions'] = accumulated_actions
            features['accumulated_sessions'] = accumulated_sessions
            features['sec_since_installation'] = (session_group.iloc[0, 2] - install_time).seconds
            features['days_since_installation'] = (session_group.iloc[0, 2] - install_time).days
            features['weekday'] = session_group.iloc[0, 2].weekday()
            features['day'] = session_group.iloc[0, 2].day
            features['hour'] = session_group.iloc[0, 2].hour

            features['total_time_mean'] = np.mean(total_times) if total_times else 0
            features['total_time_median'] = np.median(total_times) if total_times else 0
            features['total_time_max'] = np.max(total_times) if total_times else 0
            features['total_time_min'] = np.min(total_times) if total_times else 0
            features['total_time_sum'] = np.sum(total_times) if total_times else 0
            features['total_time_std'] = np.std(total_times) if len(total_times) > 1 else 0
            features['total_time_skew'] = skew(total_times) if len(total_times) > 1 else 0
            features['total_time_lag'] = total_times[-1] if len(total_times) > 0 else 0
            features['total_time_lag_diff'] = total_times[-1] - total_times[-2] if len(total_times) > 1 else 0

            features['session_activity_len_mean'] = np.mean(session_activity_lens) if session_activity_lens else 0
            features['session_activity_len_median'] = np.median(session_activity_lens) if session_activity_lens else 0
            features['session_activity_len_max'] = np.max(session_activity_lens) if session_activity_lens else 0
            features['session_activity_len_min'] = np.min(session_activity_lens) if session_activity_lens else 0
            features['session_activity_len_sum'] = np.sum(session_activity_lens) if session_activity_lens else 0
            features['session_activity_len_std'] = np.std(session_activity_lens) if len(session_activity_lens) > 1 else 0
            features['session_activity_len_skew'] = skew(session_activity_lens) if len(session_activity_lens) > 1 else 0
            features['session_activity_len_lag'] = session_activity_lens[-1] if len(session_activity_lens) > 0 else 0
            features['session_activity_len_lag_diff'] = session_activity_lens[-1] - session_activity_lens[-2] if len(session_activity_lens) > 1 else 0

            features['assess_titles_mode'] = mode(assess_titles)[0][0] if assess_titles else 'None'
            features['assess_titles_nunique'] = len(np.unique(assess_titles)) if assess_titles else 0
            features['assess_titles_lag'] = assess_titles[-1] if assess_titles else 'None'

            features['titles_mode'] = mode(titles)[0][0] if titles else 'None'
            features['titles_nunique'] = len(np.unique(titles)) if titles else 0
            features['titles_lag'] = titles[-1] if titles else 'None'

            features['current_title_count'] = custom_counter.counters['title']['title_' + session_title + '_count']
            features['current_world_count'] = custom_counter.counters['world']['world_' + session_world + '_count']
            features['current_title_total_time'] = title_times[session_title + '_title_total_time']
            features['current_world_total_time'] = world_times[session_world + '_world_total_time']
            features['current_world_game_time'] = world_game_times[session_world + '_world_game_time']
            features['current_world_activity_time'] = world_activity_times[session_world + '_world_activity_time']
            features['accumulated_misses'] = accumulated_misses
            features['all_types_accumulated_correct_attempts'] = all_types_accumulated_correct_attempts
            features['all_types_accumulated_uncorrect_attempts'] = all_types_accumulated_uncorrect_attempts
            features['change_type_count'] = change_type_count
            features.update(world_times)
            features.update(world_game_times)
            features.update(world_activity_times)
            features.update(title_times)
            features.update(weekdays)
            features.update(hours)

            all_attempts = session_group.query(f'event_code == {attempt_code}')
            true_attempts = all_attempts['event_data'].str.contains('"correct":true').sum()
            false_attempts = all_attempts['event_data'].str.contains('"correct":false').sum()
            accumulated_correct_attempts.append(true_attempts)
            accumulated_uncorrect_attempts.append(false_attempts)
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['last_accuracy_' + session_title] = accuracy
            durations.append(duration)
            accuracies.append(accuracy)
            assess_titles.append(session_title)

            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            accuracy_group_count[f"{features['accuracy_group']}_group_count"] += 1
            accumulated_accuracy_group += features['accuracy_group']
            accuracy_groups.append(features['accuracy_group'])
            last_accuracy_group_title['last_accuracy_group_' + session_title] = features['accuracy_group']

            if true_attempts + false_attempts > 0 or dataset == 'test':
                all_assessments.append(features)

            k += 1

        for count_col in config.counts:
            if count_col == 'title':
                custom_counter.counters['title']['title_' + session_title + '_count'] += 1
            elif count_col == 'world':
                custom_counter.counters['world']['world_' + session_world + '_count'] += 1
            else:
                current_counter = Counter(session_group[count_col])
                old_keys = list(current_counter.keys()).copy()
                for key in old_keys:
                    current_counter[count_col + '_' + str(key) + '_count'] = current_counter.pop(key)
                custom_counter.counters[count_col].update(current_counter)

        accumulated_actions += len(session_group)
        session_activity_lens.append(len(session_group))
        weekdays['weekday_' + str(session_group.iloc[0, 2].weekday())] += 1
        hours['hour_' + str(session_group.iloc[0, 2].hour)] += 1

        if last_type != session_type:
            change_type_count += 1
            last_type = session_type

        type_counter['type_' + session_type + '_count'] += 1
        accumulated_sessions += 1
        total_times.append(duration)
        world_times[session_world + '_world_total_time'] += duration
        title_times[session_title + '_title_total_time'] += duration
        titles.append(session_group['title'].iloc[0])

        if session_type == 'Game':
            world_game_times[session_world + '_world_game_time'] += duration
        elif session_type == 'Activity':
            world_activity_times[session_world + '_world_activity_time'] += duration

        accumulated_misses += session_group['event_data'][session_group['event_data'].str.contains('"misses"')].map(lambda x: int(json.loads(x)['misses'])).sum()
        all_types_accumulated_correct_attempts += session_group['event_data'].str.contains('"correct":true').sum()
        all_types_accumulated_uncorrect_attempts += session_group['event_data'].str.contains('"correct":false').sum()

    if dataset == 'train':
        return all_assessments
    elif dataset == 'test_for_train':
        return all_assessments[:-1]
    else:
        return [all_assessments[-1]]


def iterative_preprocessing(data, dataset, counter_path='./checkpoints/counter.pkl'):
    _data = []

    if dataset == 'train':
        custom_counter = CustomCounter(config.counts)
        custom_counter.fit(data)
        with open(counter_path, 'wb') as fout:
            pickle.dump(custom_counter.count_unique_keys, fout)
    else:
        custom_counter = CustomCounter(config.counts)
        with open(counter_path, 'rb') as fin:
            custom_counter.count_unique_keys = pickle.load(fin)

    for ins_id, ins_group in tqdm(data.groupby('installation_id', sort=False)):
        _data += preprocess_inst(ins_group, custom_counter, dataset)

    return pd.DataFrame(_data)


def add_final_features(X):
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

    count_cols = [col for col in X.columns if 'count' in col]
    for col in count_cols:
        X[col] = X[col] / X['accumulated_actions']
        X[col + '_Time_norm'] = X[col] / X['total_time_sum']

    X = X.replace({np.inf: 0})

    return X


def main(dataset='train'):
    if dataset == 'train':
        df = pd.read_csv(config.train_path)
    else:
        df = pd.read_csv(config.test_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    df = get_interaction(df, 'title', 'event_code')

    df = iterative_preprocessing(df, dataset)

    if dataset == 'train':
        cat_encoder = CategoricalEncoder(cat_cols=config.cat_cols)
        cat_encoder.fit(df)
        with open('./checkpoints/cat_encoder.pkl', 'wb') as fout:
            pickle.dump(cat_encoder, fout)
    else:
        with open('./checkpoints/cat_encoder.pkl', 'rb') as fin:
            cat_encoder = pickle.load(fin)

    df = cat_encoder.transform(df)

    df.columns = [col.replace(',', '_') if type(col) == str else str(col) for col in df.columns]

    df = df.reindex(sorted(df.columns), axis=1)

    if dataset == 'train':
        df.to_csv(config.preprocessed_train_path, index=False)
    elif dataset == 'test_for_train':
        df.to_csv(config.preprocessed_test_for_train_path, index=False)
    elif dataset == 'test':
        df.to_csv(config.preprocessed_test_path, index=False)


if __name__ == '__main__':
    main(sys.argv[1])
