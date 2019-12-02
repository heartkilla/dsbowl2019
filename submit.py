import pickle
from shutil import copyfile

import pandas as pd

copyfile(src='../input/dsbowl-offline-data/config.py', dst='../working/config.py')
copyfile(src='../input/dsbowl-offline-data/preprocessing.py', dst='../working/preprocessing.py')
copyfile(src='../input/dsbowl-offline-data/metric.py', dst='../working/metric.py')
copyfile(src='../input/dsbowl-offline-data/models.py', dst='../working/models.py')

from models import * 
from preprocessing import *

def submit():
    df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    with open('../input/dsbowl-offline-data/cat_encoder.pkl', 'rb') as fin:
        cat_encoder = pickle.load(fin)
    with open('../input/dsbowl-offline-data/model.pkl', 'rb') as fin:
        model = pickle.load(fin)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df = get_interaction(df, 'title', 'event_code')
    df = iterative_preprocessing(df, dataset='test', counter_path='../input/dsbowl-offline-data/counter.pkl')
    df = cat_encoder.transform(df)
    df.columns = [col.replace(',', '_') if type(col) == str else str(col) for col in df.columns]
    df = df.reindex(sorted(df.columns), axis=1)

    preds = model.predict(df)

    sample_submission['accuracy_group'] = preds.astype(int)
    sample_submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    sub = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    if len(sub) == 1000:
        sub.to_csv('submission.csv', index=False)
    else:
        submit()
