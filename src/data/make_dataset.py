import os

import click
import pandas as pd

from sklearn.model_selection import train_test_split

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # The data directory
DATA_DIRECTORY_RAW = os.path.join(DATA_DIRECTORY, 'raw')  # The data/raw directory
DATA_DIRECTORY_PROCESSED = os.path.join(DATA_DIRECTORY, 'processed')  # The data/processed directory


@click.group()
def cli():
    pass

def load_data_as_dataframe(filepath=None, name='raw'):
    if not filepath:
        if name == 'raw':
            filepath = os.path.join(DATA_DIRECTORY_RAW, 'data.json')
        elif name == 'train':
            filepath = os.path.join(DATA_DIRECTORY_PROCESSED, 'train.json')
        elif name == 'test':
            filepath = os.path.join(DATA_DIRECTORY_PROCESSED, 'test.json')
        else:
            print(f'no filepath exists for name {name}')
    if not os.path.exists(filepath):
        print(f"It looks like {filepath} doesn't exist. Did you unzip the main data?")
        return None

    df = pd.read_json(filepath)
    df = df.sort_index()

    return df

@cli.command()
@click.option('--numrows', default=1000, type=int, help='How many rows to create in the data subset.')
@click.option('--outfile', default=None, type=str, help='Where to save the subset file.')
@click.option('--file-type', default='json', help='The type of file to save the subset as.')
@click.option('--random-subset', is_flag=True, default=False, help='Returns a random subset of data, rather than the first rows.')
def create_subset(numrows, outfile, file_type, random_subset):
    df = load_data_as_dataframe()
    df_subset = df[:numrows]

    if not outfile:
        outfile = os.path.join(DATA_DIRECTORY_PROCESSED, f'data-{numrows}.{file_type}')

    if file_type == 'json':
        df_subset.to_json(outfile)
    elif file_type == 'csv':
        df_subset.to_csv(outfile, index=False)
    else:
        print("It looks like the file_type you entered isn't supported.")

@cli.command()
def make_train_test_split():
    train_outfile = os.path.join(DATA_DIRECTORY_PROCESSED, f'train.json')
    test_outfile = os.path.join(DATA_DIRECTORY_PROCESSED, f'test.json')
    if os.path.exists(train_outfile) and os.path.exists(test_outfile):
        print('both files already exists, not doing anything')
        return
    print('loading data')
    df = load_data_as_dataframe()
    print('creating target column for stratified train/test split')
    df['fraud'] = df['acct_type'].str.contains('fraud').values.astype(int)
    df.drop(columns='acct_type', inplace=True)
    print('creating train test split')
    labels_array = df['fraud'].to_numpy()
    train, test = train_test_split(df, random_state=42, stratify=labels_array)
    print(f'writing train file to {train_outfile}')
    train.to_json(train_outfile)
    print(f'writing test file to {test_outfile}')
    test.to_json(test_outfile)


if __name__ == "__main__":
    cli()
