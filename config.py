train_path = './data/train.csv'
test_path = './data/test.csv'
train_labels_path = './data/train_labels.csv'
specs_path = './data/specs.csv'
sample_sub_path = './data/sample_submission.csv'
preprocessed_train_path = './data/preprocessed_train.csv'
preprocessed_test_path = './data/preprocessed_test.csv'

# no different categories in train/test for now but be careful
cat_cols = ['title', 'world']

counts = ['event_code', 'world', 'title', 'title_event_code']

n_folds = 5

lgb_params = {'objective': 'huber',
              'alpha': 2.5,
              'boosting': 'gbdt',
              'metric': 'None',
              'num_leaves': 32,
              'max_depth': 7,
              'min_data_in_leaf': 80,
              'learning_rate': 0.01,
              'bagging_fraction': 0.5,
              'bagging_freq': 1,
              'feature_fraction': 0.3,
              'feature_fraction_seed': 44,
              'lambda_l1': 0,
              'lambda_l2': 1,
              'verbosity': -1,
              'first_metric_only': True,
              'seed': 271828}

lgb_train_params = {'num_boost_round': 1000000,
                    'early_stopping_rounds': 500,
                    'verbose_eval': 100}


"""
Train mean QWK: 0.741730+/-0.010295
CV mean QWK: 0.599483+/-0.016011
CV OOF QWK: 0.599560
CV random QWK: 0.567528+/-0.028369
CV OOF random QWK: 0.567967


0.567814
"""
