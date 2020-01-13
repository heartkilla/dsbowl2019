train_path = './data/train.csv'
test_path = './data/test.csv'
train_labels_path = './data/train_labels.csv'
specs_path = './data/specs.csv'
sample_sub_path = './data/sample_submission.csv'
preprocessed_train_path = './data/preprocessed_train.csv'
preprocessed_test_path = './data/preprocessed_test.csv'
preprocessed_test_for_train_path = './data/preprocessed_test_for_train.csv'

# no different categories in train/test for now but be careful
cat_cols = ['title', 'world', 'assess_titles_mode', 'assess_titles_lag', 'titles_mode', 'titles_lag']

counts = ['event_code', 'world', 'title', 'title_event_code']

n_folds = 5

lgb_params = {'objective': 'mse',
              #'alpha': 2.5,
              'boosting': 'gbdt',
              'metric': 'None',
              'num_leaves': 16,
              'max_depth': 5,
              'min_data_in_leaf': 100,
              'learning_rate': 0.01,
              'bagging_fraction': 0.35,
              'bagging_freq': 1,
              'feature_fraction': 0.3,
              'feature_fraction_seed': 35,
              'lambda_l1': 0,
              'lambda_l2': 0.1,
              'verbosity': -1,
              'first_metric_only': True,
              'seed': 49}

# CV random QWK:  0.564056+/-0.024703

lgb_train_params = {'num_boost_round': 1000000,
                    'early_stopping_rounds': 500,
                    'verbose_eval': 100,
                    'cat_feats': ['assess_titles_mode']}

entire_train_stats = False
