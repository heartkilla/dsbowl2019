import pickle

import pandas as pd

import config
from metric import allocate_to_rate
from preprocessing import add_final_features


if __name__ == '__main__':
    X = pd.read_csv(config.preprocessed_test_path)
    sample_submission = pd.read_csv(config.sample_sub_path)

    with open('./checkpoints/model_1.pkl', 'rb') as fin:
        model_1 = pickle.load(fin)
    with open('./checkpoints/model_2.pkl', 'rb') as fin:
        model_2 = pickle.load(fin)
    with open('./checkpoints/model_3.pkl', 'rb') as fin:
        model_3 = pickle.load(fin)
    with open('./checkpoints/model_1_new.pkl', 'rb') as fin:
        model_1_new = pickle.load(fin)
    with open('./checkpoints/model_3_new.pkl', 'rb') as fin:
        model_3_new = pickle.load(fin)
    with open('./checkpoints/stacker.pkl', 'rb') as fin:
        stacker = pickle.load(fin)

    X = add_final_features(X)

    X['feat_1'] = allocate_to_rate(model_1.predict(X, raw_values=True))
    X['feat_2'] = allocate_to_rate(model_2.predict(X, raw_values=True))
    X['feat_3'] = allocate_to_rate(model_3.predict(X, raw_values=True))
    X['feat_1_new'] = allocate_to_rate(model_1_new.predict(X, raw_values=True))
    X['feat_3_new'] = allocate_to_rate(model_3_new.predict(X, raw_values=True))

    preds = stacker.predict(X, raw_values=False)

    sample_submission['accuracy_group'] = preds.astype(int)
    sample_submission.to_csv('./data/submission.csv', index=False)

    print(sample_submission['accuracy_group'].value_counts(normalize=True))
