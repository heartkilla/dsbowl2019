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
from scipy.stats import skew
from datetime import timedelta

import config

assessments = ['Cart Balancer (Assessment)',
               'Cauldron Filler (Assessment)',
               'Chest Sorter (Assessment)',
               'Mushroom Sorter (Assessment)',
               'Bird Measurer (Assessment)']


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
        self.counters = {
        count_col: Counter({count_col + '_' + str(key) + '_count': 0 for key in self.count_unique_keys[count_col]}) for
        count_col in self.count_unique_keys}


def get_interaction(data, attr_1, attr_2):
    data[attr_1 + '_' + attr_2] = data[attr_1].astype(str) + '_' + data[attr_2].astype(str)
    return data


def get_accuracy_group(accuracy):
    if accuracy == 0:
        return 0
    elif accuracy == 1:
        return 3
    elif accuracy == 0.5:
        return 2
    else:
        return 1

def preprocess_assessments_data(assessment_data):
    preprocessed_assessments = []
    for game_session, session_group in assessment_data.groupby("game_session", sort=False):
        result = {}
        result["assessment_name"] = session_group.iloc[0]["title"]
        result["installation_id"] = session_group.iloc[0]["installation_id"]
        if result["assessment_name"] == 'Bird Measurer (Assessment)':
            submission_code = 4110
        else:
            submission_code = 4100
        duration = session_group.iloc[-1]["timestamp"] - session_group.iloc[0]["timestamp"]
        result["duration_seconds"] = duration.seconds
        submissions = session_group.query(f'event_code == {submission_code}')
        result["submissions_number"] = len(submissions)
        result["is_success"] = any(submissions["event_data"].str.contains('"correct":true'))
        result["accuracy"] = result["is_success"] / result["submissions_number"] if result["submissions_number"] > 0 else 0
        result["accuracy_group"] = get_accuracy_group(result["accuracy"])
        result["events_number"] = len(session_group)
        result["date"] = session_group.iloc[-1]["timestamp"].date()
        result["submitted"] = result["submissions_number"] > 0
        preprocessed_assessments.append(result)
    return pd.DataFrame(preprocessed_assessments).sort_values("date")


def get_assessment_daily_statistics(preprocessed_assessments):
    start_date = preprocessed_assessments["date"].min()
    end_date = preprocessed_assessments["date"].max()
    assessments_statistics = []
    for assessment in assessments:
        assessment_data = preprocessed_assessments[preprocessed_assessments["assessment_name"] == assessment]
        for date in pd.date_range(start=start_date, end=end_date):
            submitted_assessment_by_day = assessment_data[(assessment_data["date"] < date) & (assessment_data["submitted"])]
            all_assessments_by_day = assessment_data[(assessment_data["date"] < date)]
            result = {"date": date,
                      "assessment_name": assessment,
                      # 1 day is substracted to take the previous day before the current
                      "that_day_submitted_assessments": len(submitted_assessment_by_day[submitted_assessment_by_day["date"]
                                            == (date - timedelta(days=1))]),
                      "that_day_unique_users": submitted_assessment_by_day[submitted_assessment_by_day["date"]
                                            == (date - timedelta(days=1))]["installation_id"].nunique(),
                      "duration_mean": np.mean(submitted_assessment_by_day["duration_seconds"]),
                      "duration_median": np.median(submitted_assessment_by_day["duration_seconds"]),
                      "duration_std": np.std(submitted_assessment_by_day["duration_seconds"]),
                      "submissions_number_mean": np.mean(submitted_assessment_by_day["submissions_number"]),
                      "submissions_number_std": np.std(submitted_assessment_by_day["submissions_number"]),
                      "success_mean": np.mean(submitted_assessment_by_day["is_success"]),
                      "success_std": np.std(submitted_assessment_by_day["is_success"]),
                      "accuracy_mean": np.mean(submitted_assessment_by_day["accuracy"]),
                      "accuracy_median": np.median(submitted_assessment_by_day["accuracy"]),
                      "accuracy_std": np.std(submitted_assessment_by_day["accuracy"]),
                      "event_number_mean": np.mean(submitted_assessment_by_day["events_number"]),
                      "events_number_median": np.median(submitted_assessment_by_day["events_number"]),
                      "events_number_std": np.std(submitted_assessment_by_day["events_number"]),
                      "accuracy_group_1_rate": len(submitted_assessment_by_day["accuracy_group"] == 1) \
                                               / len(submitted_assessment_by_day) if len(submitted_assessment_by_day) > 0 else 0,
                      "accuracy_group_2_rate": len(submitted_assessment_by_day["accuracy_group"] == 2) \
                                               / len(submitted_assessment_by_day) if len(submitted_assessment_by_day) > 0 else 0,
                      "accuracy_group_3_rate": len(submitted_assessment_by_day["accuracy_group"] == 3) \
                                               / len(submitted_assessment_by_day) if len(submitted_assessment_by_day) > 0 else 0,
                      "accuracy_group_4_rate": len(submitted_assessment_by_day["accuracy_group"] == 4) \
                                               / len(submitted_assessment_by_day) if len(submitted_assessment_by_day) > 0 else 0,
                      "submission_rate": len(submitted_assessment_by_day) / len(all_assessments_by_day) if len(all_assessments_by_day) > 0 else 0,
                      # 1 day is substracted to take the previous day before the current
                      "that_day_all_assessments": len(all_assessments_by_day[all_assessments_by_day["date"]
                                                                             == (date - timedelta(days=1))])}
            assessments_statistics.append(result)
    return pd.DataFrame(assessments_statistics)

def preprocess_inst(ins_group, custom_counter, dataset):
    all_assessments = []

    custom_counter.reset()

    types = ['Clip', 'Activity', 'Assessment', 'Game']
    # changed_type_counter = {'changed_type_' + col + '_count': 0 for col in types}
    type_counter = {'type_' + col + '_count': 0 for col in types}
    last_type = 0

    accumulated_correct_attempts = []
    accumulated_uncorrect_attempts = 0
    accumulated_accuracy = 0

    last_accuracy_title = {'last_accuracy_' + title: np.nan for title in assessments}
    accuracy_groups = {f'{i}_group_count': 0 for i in range(4)}
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
            # features.update(changed_type_counter)
            features.update(type_counter)
            features['assessment_date'] = session_group['timestamp'].iloc[0].date()
            features['assessment_title'] = session_title
            features['accumulated_correct_attempts_mean'] = np.mean(
                accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_median'] = np.median(
                accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_sum'] = np.sum(
                accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_max'] = np.max(
                accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_min'] = np.min(
                accumulated_correct_attempts) if accumulated_correct_attempts else 0
            features['accumulated_correct_attempts_std'] = np.std(accumulated_correct_attempts) if len(
                accumulated_correct_attempts) > 1 else 0
            features['accumulated_correct_attempts_skew'] = skew(accumulated_correct_attempts) if len(
                accumulated_correct_attempts) > 1 else 0

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            features['accumulated_accuracy'] = accumulated_accuracy / k if k > 0 else 0
            features.update(last_accuracy_title)

            features['assess_duration_mean'] = np.mean(durations) if durations else 0
            features['assess_duration_median'] = np.median(durations) if durations else 0
            features['assess_duration_max'] = np.max(durations) if durations else 0
            features['assess_duration_min'] = np.min(durations) if durations else 0
            features['assess_duration_sum'] = np.sum(durations) if durations else 0
            features['assess_duration_std'] = np.std(durations) if len(durations) > 1 else 0
            features['assess_duration_skew'] = skew(durations) if len(durations) > 1 else 0

            features['accuracies_mean'] = np.mean(accuracies) if accuracies else 0
            features['accuracies_median'] = np.median(accuracies) if accuracies else 0
            features['accuracies_max'] = np.max(accuracies) if accuracies else 0
            features['accuracies_min'] = np.min(accuracies) if accuracies else 0
            features['accuracies_sum'] = np.sum(accuracies) if accuracies else 0
            features['accuracies_std'] = np.std(accuracies) if len(accuracies) > 1 else 0
            features['accuracies_skew'] = skew(accuracies) if len(accuracies) > 1 else 0

            features.update(accuracy_groups)
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

            all_attempts = session_group.query(f'event_code == {attempt_code}')
            true_attempts = all_attempts['event_data'].str.contains('"correct":true').sum()
            false_attempts = all_attempts['event_data'].str.contains('"correct":false').sum()
            accumulated_correct_attempts.append(true_attempts)
            accumulated_uncorrect_attempts += false_attempts
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['last_accuracy_' + session_title] = accuracy
            durations.append(duration)
            accuracies.append(accuracy)

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

            if true_attempts + false_attempts > 0 or dataset != 'train':
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

        if last_type != session_type:
            change_type_count += 1
            last_type = session_type

        type_counter['type_' + session_type + '_count'] += 1
        accumulated_sessions += 1
        total_times.append(duration)
        world_times[session_world + '_world_total_time'] += duration
        title_times[session_title + '_title_total_time'] += duration

        if session_type == 'Game':
            world_game_times[session_world + '_world_game_time'] += duration
        elif session_type == 'Activity':
            world_activity_times[session_world + '_world_activity_time'] += duration

        accumulated_misses += session_group['event_data'][session_group['event_data'].str.contains('"misses"')].map(
            lambda x: int(json.loads(x)['misses'])).sum()
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


def main(dataset='train'):
    if dataset == 'train':
        df = pd.read_csv(config.train_path)
        cat_encoder = CategoricalEncoder(cat_cols=config.cat_cols)
        cat_encoder.fit(df)
        with open('./checkpoints/cat_encoder.pkl', 'wb') as fout:
            pickle.dump(cat_encoder, fout)
    elif dataset == 'assessment':
        train_df = pd.read_csv(config.train_path)
        assessment_data = train_df[train_df["type"] == "Assessment"].copy(deep=True)
        del train_df
        test_df = pd.read_csv(config.test_path)
        assessment_data = pd.concat([assessment_data, test_df[test_df["type"] == "Assessment"].copy(deep=True)])
        del test_df
        assessment_data['timestamp'] = pd.to_datetime(assessment_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
        preprocessed_assessments_data = preprocess_assessments_data(assessment_data)
        assessment_daily_statistics = get_assessment_daily_statistics(preprocessed_assessments_data)
        assessment_daily_statistics.to_csv(config.assessment_path, index=False)
        return
    elif dataset == 'merge':
        preprocessed_train_df = pd.read_csv(config.preprocessed_train_path)
        preprocessed_train_df['assessment_date'] = \
            pd.to_datetime(preprocessed_train_df['assessment_date'], format='%Y-%m-%d')
        assessments = pd.read_csv(config.assessment_path)
        assessments['date'] = pd.to_datetime(assessments['date'], format='%Y-%m-%d')
        assessments.columns = ["".join(c if c.isalnum()
                                                 else "_" for c in str(x)) for x in assessments.columns]
        assessments["date"] -= pd.Timedelta(days=1)
        preprocessed_train_df = pd.merge(preprocessed_train_df, assessments, how='left',
                                         left_on=["assessment_title", "assessment_date"],
                                         right_on=["assessment_name", "date"])
        preprocessed_train_df = preprocessed_train_df.drop\
            (columns=["assessment_name", "date", "assessment_title", "assessment_date"])
        preprocessed_train_df.to_csv(config.preprocessed_train_path)

        test_for_train = pd.read_csv(config.preprocessed_test_for_train_path)
        test_for_train['assessment_date'] = \
            pd.to_datetime(test_for_train['assessment_date'], format='%Y-%m-%d')
        test_for_train = pd.merge(test_for_train, assessments, how='left',
                                         left_on=["assessment_title", "assessment_date"],
                                         right_on=["assessment_name", "date"])
        test_for_train = test_for_train.drop \
            (columns=["assessment_name", "date", "assessment_title", "assessment_date"])
        test_for_train.to_csv(config.preprocessed_test_for_train_path)
        return
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

    if dataset == 'train':
        df.to_csv(config.preprocessed_train_path, index=False)
    elif dataset == 'test_for_train':
        df.to_csv(config.preprocessed_test_for_train_path, index=False)
    elif dataset == 'test':
        df.to_csv(config.preprocessed_test_path, index=False)


if __name__ == '__main__':
    main(sys.argv[1])
