import pandas as pandas
import numpy as np
import math


class Kfcv:
    def __init__(self, data_set, k_input):

        try:
            # Reads the csv file using Pandas library and save it as data frame
            if data_set == 'ecoli':
                records = pandas.read_csv('data/ecoli.csv', encoding="ISO-8859-1")
                records.rename(columns={'locsite': 'class'}, inplace=True)
            elif data_set == 'glass':
                records = pandas.read_csv('data/glass.csv', encoding="ISO-8859-1")
            elif data_set == 'yeast':
                records = pandas.read_csv('data/yeast.csv', encoding="ISO-8859-1")
            else:
                raise Exception('Please enter proper data set')

            # Shuffling the Data frame randomly instead of randomly picking one records from frame
            # Discussed with professor and he agreed for this randomness way
            records = records.iloc[np.random.permutation(len(records))]
            records = records.reset_index(drop=True)

            pandas.DataFrame(records).to_csv('output/random_shuffle.csv', index=False)
            # Finding min and max value for each column using pandas id max function on column
            for column in records.columns:
                if column != 'class' and column != 'Id' and column != 'Sequence':
                    column_max = records.loc[records[column].idxmax()][column]
                    column_min = records.loc[records[column].idxmin()][column]
                    records[column] = records[column].map(lambda x: (x - column_min) / (column_max - column_min))
                    records[column] = records[column].astype('float64');

            # Saving the data fram as output csv file for reference.
            pandas.DataFrame(records).to_csv('output/result.csv', index=False)

            # Splitting the records set into partitions
            partitions = Kfcv.split(records, math.ceil(len(records) / k_input))
            self.partitions = partitions

        except Exception as error:
            print("problem while processing the files", repr(error))
            self.partitions = None

    def get_partitions(self):
        return self.partitions

    @staticmethod
    # Method to Split the List into K size of partitions
    def split(iterable, n):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
