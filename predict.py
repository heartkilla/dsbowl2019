import pickle

import pandas as pd

import config


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_test_path)
    sample_submission = pd.read_csv(config.sample_sub_path)

    with open('./checkpoints/model.pkl', 'rb') as fin:
        model = pickle.load(fin)

    count_cols = [col for col in X.columns if 'count' in col]
    for col in count_cols:
        X[col] = X[col] / X['accumulated_actions']

    X['accuracy'] = 0

    preds = model.predict(X)

    sample_submission['accuracy_group'] = preds.astype(int)
    sample_submission.to_csv('./data/submission.csv', index=False)

    print(sample_submission['accuracy_group'].value_counts(normalize=True))
