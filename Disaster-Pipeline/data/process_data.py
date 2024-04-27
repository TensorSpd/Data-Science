import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from CSV files and merge them into a single DataFrame.

    Parameters:
    messages_filepath (str): Filepath to the CSV file containing messages data.
    categories_filepath (str): Filepath to the CSV file containing categories data.

    Returns:
    DataFrame: A DataFrame containing merged data from messages and categories.
    """
    # Load messages data from CSV file
    messages = pd.read_csv(messages_filepath)

    # Load categories data from CSV file
    categories = pd.read_csv(categories_filepath)

    # Merge messages and categories data on 'id' column
    df = messages.merge(categories, how='inner', on='id')

    return df


def clean_data(df):
    """
    Cleans the input dataframe by splitting the 'categories' column into separate category columns,
    converting values to numeric, and removing duplicates.

    Args:
        df (DataFrame): Input dataframe containing the 'categories' column.

    Returns:
        DataFrame: Cleaned dataframe with individual category columns.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.loc[0]

    # rename column names after truncating the string
    col_names = row.apply(lambda x: x[:-2])
    categories.columns = col_names

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save DataFrame to a SQLite database.

    Args:
        df (DataFrame): DataFrame to be saved.
        database_filename (str): Name of the SQLite database file.

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
