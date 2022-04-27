import glob

import pandas as pd
from natsort import natsorted

csv_files = natsorted(glob.glob('*.csv'))

print("file", "precision", "accuracy", "recall", "f1")
for file in csv_files:
    # if 'mutants' in file and ('1100' in file or 'all' in file):
    # if 'all' in file:

    df = pd.read_csv(file)
    # df = df.loc[df['aggregation_type'].str.contains('max')]

    print(file,
          round(df["precision"].mean()),
          round(df["accuracy"].mean()),
          round(df["recall"].mean()),
          round(df["f1"].mean()))
