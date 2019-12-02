import copy
import pickle
from collections import Counter
from shutil import copyfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

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
    type_counter = {col + '_count': 0 for col in types}
    last_type = 0

    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_accuracy = 0
    assessments = ['Cart Balancer (Assessment)',
                   'Cauldron Filler (Assessment)',
                   'Chest Sorter (Assessment)',
                   'Mushroom Sorter (Assessment)',
                   'Bird Measurer (Assessment)']
    last_accuracy_title = {'acc_' + title: -1 for title in assessments}
    accuracy_groups = {f'{i}_group_count': 0 for i in range(4)}
    accumulated_accuracy_group = 0
    accumulated_actions = 0

    durations = []

    k = 0
    for game_session, session_group in ins_group.groupby('game_session',
                                                         sort=False):
        session_inst_id = session_group['installation_id'].iloc[0]
        session_type = session_group['type'].iloc[0]
        session_title = session_group['title'].iloc[0]
        session_world = session_group['world'].iloc[0]

        if session_title == 'Bird Measurer (Assessment)':
            attempt_code = 4110
        else:
            attempt_code = 4100

        if session_type == 'Assessment' and (dataset == 'test' or len(session_group) > 1):
            features = {}
            features['installation_id'] = session_inst_id
            features['title'] = session_title
            features['world'] = session_world
            for counter in custom_counter.counters.values():
                features.update(counter)
            features.update(type_counter)
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            features['accumulated_accuracy'] = accumulated_accuracy / k if k > 0 else 0
            features.update(last_accuracy_title)
            features['duration_mean'] = np.mean(durations) if durations else 0
            features.update(accuracy_groups)
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / k if k > 0 else 0
            features['accumulated_actions'] = accumulated_actions

            all_attempts = session_group.query(f'event_code == {attempt_code}')
            true_attempts = all_attempts['event_data'].str.contains('"correct":true').sum()
            false_attempts = all_attempts['event_data'].str.contains('"correct":false').sum()
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title] = accuracy
            durations.append((session_group.iloc[-1, 2] - session_group.iloc[0, 2]).seconds)

            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            accuracy_groups[f"{features['accuracy_group']}_group_count"] += 1
            accumulated_accuracy_group += features['accuracy_group']

            if true_attempts + false_attempts > 0 or dataset == 'test':
                all_assessments.append(features)

            k += 1

        for count_col in config.counts:
            current_counter = Counter(session_group[count_col])
            old_keys = list(current_counter.keys()).copy()
            for key in old_keys:
                current_counter[count_col + '_' + str(key) + '_count'] = current_counter.pop(key)
            custom_counter.counters[count_col].update(current_counter)

        accumulated_actions += len(session_group)

        if last_type != session_type:
            type_counter[session_type + '_count'] += 1
            last_type = session_type

    return all_assessments if dataset == 'train' else [all_assessments[-1]]


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


def main(dataset='train'):
    if dataset == 'train':
        df = pd.read_csv(config.train_path)
        cat_encoder = CategoricalEncoder(cat_cols=config.cat_cols)
        cat_encoder.fit(df)
        with open('./checkpoints/cat_encoder.pkl', 'wb') as fout:
            pickle.dump(cat_encoder, fout)
    else:
        df = pd.read_csv(config.test_path)
        with open('./checkpoints/cat_encoder.pkl', 'rb') as fin:
            cat_encoder = pickle.load(fin)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    df = get_interaction(df, 'title', 'event_code')

    df = iterative_preprocessing(df, dataset)

    df = cat_encoder.transform(df)

    df.columns = [col.replace(',', '_') if type(col) == str else str(col) for col in df.columns]

    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv(f'./data/preprocessed_{dataset}.csv', index=False)


if __name__ == '__main__':
    main('test')
