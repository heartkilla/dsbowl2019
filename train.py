import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config
from models import LGBMModel


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_train_path)
    X_add = pd.read_csv(config.preprocessed_test_for_train_path)
    X = pd.concat([X, X_add])

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

    print(X.shape)

    y = X['accuracy_group']

    X = X[['accuracy_group', 'installation_id',
    'title',
 'titles_lag',
 'world',
 'event_code_3121_count',
 'event_code_3021_count',
 'current_title_count',
 'assess_titles_lag',
 'event_code_4020_count',
 'accuracies_mean',
 'current_world_mean_time',
 'event_code_4025_count',
 'event_code_4030_count',
 'event_code_4090_count',
 'event_code_2010_count',
 'accumulated_accuracy',
 'event_code_3010_count',
 'all_types_accumulated_uncorrect_attempts',
 'accumulated_accuracy_group',
 'title_event_code_Sandcastle Builder (Activity)_4020_count',
 'total_time_lag',
 'time_per_event',
 'session_activity_len_lag_diff',
 'total_time_lag_diff',
 'accuracies_std',
 'event_code_2010_count_Time_norm',
 'day_sin',
 'accuracies_median',
 'assess_duration_lag',
 'event_code_3120_count_Time_norm',
 'event_code_3020_count_Time_norm',
 'accuracy_groups_std',
 'title_Crystal Caves - Level 3_count_Time_norm',
 'current_title_total_time',
 'event_code_4035_count_Time_norm',
 'assess_duration_min',
 'event_code_4025_count_Time_norm',
 'accumulated_misses',
 'title_event_code_Cauldron Filler (Assessment)_4025_count',
 'title_event_code_Happy Camel_3120_count',
 'world_TREETOPCITY_count',
 'world_CRYSTALCAVES_count',
 'title_Crystal Caves - Level 2_count',
 'accumulated_uncorrect_attempts_mean',
 'current_world_total_time',
 'event_code_4020_count_Time_norm',
 'title_event_code_Cart Balancer (Assessment)_4070_count',
 'title_event_code_Bottle Filler (Activity)_4070_count',
 'title_Welcome to Lost Lagoon!_count_Time_norm',
 'ratio_of_life_in_game',
 'title_event_code_Happy Camel_4070_count',
 'title_Crystal Caves - Level 1_count',
 'title_event_code_Sandcastle Builder (Activity)_4035_count',
 'accumulated_uncorrect_attempts_std',
 'accumulated_sessions',
 'hour_cos',
 'event_code_4022_count',
 'event_code_4030_count_Time_norm',
 'title_Magma Peak - Level 1_count',
 'event_code_4022_count_Time_norm',
 'last_accuracy_Bird Measurer (Assessment)',
 'assess_duration_lag_diff',
 'title_event_code_Pan Balance_3020_count',
 'accumulated_correct_attempts_mean',
 'title_event_code_All Star Sorting_3120_count',
 'last_accuracy_Cauldron Filler (Assessment)',
 'accuracies_sum',
 'title_event_code_Crystal Caves - Level 3_2000_count_Time_norm',
 'event_code_2060_count',
 'type_Game_count_Time_norm',
 'assess_duration_median',
 'title_event_code_Bug Measurer (Activity)_4035_count',
 '0_group_count',
 'last_accuracy_Chest Sorter (Assessment)',
 'event_code_4040_count_Time_norm',
 'title_event_code_Chow Time_3121_count',
 'event_code_4100_count_Time_norm',
 'title_Magma Peak - Level 1_count_Time_norm',
 'total_time_max',
 'event_code_2080_count',
 'sec_since_installation',
 'title_event_code_Chest Sorter (Assessment)_4025_count',
 'title_event_code_Egg Dropper (Activity)_4070_count_Time_norm',
 'title_event_code_Chest Sorter (Assessment)_4020_count',
 'title_Tree Top City - Level 1_count_Time_norm',
 'type_Clip_count',
 'title_event_code_Cart Balancer (Assessment)_4070_count_Time_norm',
 'type_Activity_count_Time_norm',
 'world_MAGMAPEAK_count_Time_norm',
 'title_Ordering Spheres_count',
 'title_Magma Peak - Level 2_count_Time_norm',
 'total_time_std',
 'event_code_2000_count',
 'title_event_code_Tree Top City - Level 1_2000_count',
 'title_event_code_Welcome to Lost Lagoon!_2000_count',
 'title_event_code_Sandcastle Builder (Activity)_4070_count',
 'title_event_code_Sandcastle Builder (Activity)_4035_count_Time_norm',
 'world_CRYSTALCAVES_count_Time_norm',
 'Sandcastle Builder (Activity)_title_total_time',
 'title_event_code_Tree Top City - Level 3_2000_count',
 'title_event_code_All Star Sorting_2025_count',
 'title_event_code_Tree Top City - Level 2_2000_count',
 'title_event_code_Egg Dropper (Activity)_4070_count',
 'title_Crystal Caves - Level 2_count_Time_norm',
 'title_event_code_Sandcastle Builder (Activity)_3010_count',
 'event_code_4045_count',
 'title_event_code_Crystal Caves - Level 2_2000_count',
 'total_time_mean',
 'title_event_code_Cauldron Filler (Assessment)_4070_count',
 'title_event_code_Chow Time_3021_count',
 'accuracies_lag_diff',
 'title_event_code_All Star Sorting_2030_count',
 'title_Bottle Filler (Activity)_count',
 'title_event_code_Chest Sorter (Assessment)_3021_count',
 'event_code_2035_count',
 'title_event_code_Chest Sorter (Assessment)_4030_count',
 'session_activity_len_mean',
 'title_event_code_Fireworks (Activity)_4070_count',
 'event_code_4021_count',
 'title_event_code_Welcome to Lost Lagoon!_2000_count_Time_norm',
 'title_event_code_Scrub-A-Dub_4010_count',
 'event_code_4010_count_Time_norm',
 'title_Crystal Caves - Level 1_count_Time_norm',
 'all_types_accumulated_correct_attempts',
 'event_code_2080_count_Time_norm',
 'title_event_code_Sandcastle Builder (Activity)_4021_count',
 'title_Lifting Heavy Things_count',
 'title_event_code_Pan Balance_3120_count',
 'type_Assessment_count',
 'title_event_code_Fireworks (Activity)_3010_count',
 'event_code_2020_count_Time_norm',
 'title_event_code_Magma Peak - Level 2_2000_count',
 'title_event_code_Flower Waterer (Activity)_4070_count',
 'time_per_session',
 'title_event_code_Fireworks (Activity)_3110_count',
 '3_group_count_Time_norm',
 'title_event_code_Chow Time_2030_count',
 'title_event_code_Egg Dropper (Activity)_4020_count',
 'title_event_code_Leaf Leader_2030_count',
 'title_Scrub-A-Dub_count',
 'title_event_code_Watering Hole (Activity)_4070_count',
 'event_code_4070_count_Time_norm',
 'event_code_3110_count_Time_norm',
 'title_Tree Top City - Level 3_count_Time_norm',
 'accuracies_lag',
 'title_Egg Dropper (Activity)_count',
 'title_event_code_All Star Sorting_4035_count',
 'event_code_2083_count',
 'title_Sandcastle Builder (Activity)_count',
 'title_event_code_Sandcastle Builder (Activity)_4030_count',
 'event_code_3010_count_Time_norm',
 'event_code_2075_count',
 'days_since_installation',
 'assess_duration_max',
 'title_event_code_Crystal Caves - Level 2_2000_count_Time_norm',
 'title_Tree Top City - Level 2_count_Time_norm',
 'title_event_code_Crystal Caves - Level 1_2000_count',
 'title_event_code_Fireworks (Activity)_4030_count',
 'title_event_code_Bottle Filler (Activity)_4070_count_Time_norm',
 'title_event_code_Chest Sorter (Assessment)_4020_count_Time_norm',
 '0_group_count_Time_norm',
 'title_event_code_Scrub-A-Dub_2000_count',
 'title_event_code_Chest Sorter (Assessment)_3121_count',
 'title_event_code_Tree Top City - Level 1_2000_count_Time_norm',
 'events_per_session',
 'title_event_code_Fireworks (Activity)_4020_count',
 'title_event_code_Crystals Rule_3120_count',
 'event_code_2083_count_Time_norm',
 'event_code_4021_count_Time_norm',
 'title_Balancing Act_count',
 'MAGMAPEAK_world_activity_time',
 'title_event_code_Sandcastle Builder (Activity)_3110_count',
 'title_event_code_Chow Time_4035_count',
 'CRYSTALCAVES_world_game_time',
 'title_event_code_Magma Peak - Level 1_2000_count',
 'title_event_code_Air Show_3021_count',
 'title_Slop Problem_count_Time_norm',
 'title_event_code_Leaf Leader_3021_count',
 'Cart Balancer (Assessment)_title_total_time',
 'event_code_2060_count_Time_norm',
 'weekday_sin',
 'world_TREETOPCITY_count_Time_norm',
 'title_event_code_Magma Peak - Level 2_2000_count_Time_norm',
 'CRYSTALCAVES_world_total_time',
 'title_event_code_Cauldron Filler (Assessment)_4100_count',
 'event_code_4095_count',
 'title_event_code_Cart Balancer (Assessment)_4020_count',
 'assess_duration_std',
 'title_event_code_Crystals Rule_3020_count',
 'title_event_code_Cart Balancer (Assessment)_4100_count',
 'title_event_code_Chow Time_4070_count_Time_norm',
 'Bottle Filler (Activity)_title_total_time',
 'title_event_code_Bubble Bath_3020_count',
 'current_world_count_Time_norm',
 'world_NONE_count_Time_norm',
 'title_event_code_Tree Top City - Level 2_2000_count_Time_norm',
 'title_event_code_Happy Camel_4035_count',
 'title_event_code_Bottle Filler (Activity)_4035_count',
 'title_event_code_Ordering Spheres_2000_count',
 'title_event_code_Magma Peak - Level 1_2000_count_Time_norm',
 'title_event_code_Mushroom Sorter (Assessment)_4070_count']]

    model = LGBMModel(params=config.lgb_params,
                      folds=GroupKFold(n_splits=config.n_folds),
                      cols_to_drop=['installation_id', 'accuracy_group'],
                      group_col='installation_id',
                      **config.lgb_train_params)

    model.fit(X, y)

    with open('./checkpoints/model.pkl', 'wb') as fout:
            pickle.dump(model, fout)
