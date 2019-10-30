import os

import click
import pandas as pd

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # The data directory
DATA_DIRECTORY_RAW = os.path.join(DATA_DIRECTORY, 'raw')  # The data/raw directory
DATA_DIRECTORY_PROCESSED = os.path.join(DATA_DIRECTORY, 'processed')  # The data/processed directory


@click.group()
def cli():
    pass

def load_data_as_dataframe(filepath=None):
    if not filepath:
        filepath = os.path.join(DATA_DIRECTORY_RAW, 'data.json')
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


if __name__ == "__main__":
    cli()
