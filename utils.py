from mldata import *
import pandas as pd
import numpy as np
from copy import deepcopy
import math
import random

type_key = 'type'
choices_key = 'choices'
CLASS_NAME = 'CLASS'


def data_to_dataframe(data):
    '''
    This function creates a Pandas Dataframe out of the parsed c45 file. We also drop the
    first column which is typically 'id' to prevent identity attributes from affecting id3
    '''
    dataframe = pd.DataFrame(data)

    names = []
    for col in range(len(data[0])):
        names.append(data.schema[col].name)

    dataframe.columns = names
    dataframe.drop(dataframe.columns[0], axis=1,
                   inplace=True)  # Drop id column

    return dataframe


def process_data(df, attr_dict):
    """
    Turn nominal attributes to numbers and then normalize dataset

    :param DataFrame df: The dataframe to process
    :param dict attr_dict: The attribute dictionary
    :return: Processed dataframe, mapping list to unmap nominal attributes
    """

    map_list = []
    processed_df = df
    for attr in attr_dict:
        if attr_type(attr_dict, attr) == Feature.Type.NOMINAL:
            unique_vals = attr_choices(attr_dict, attr)
            mapped_vals = dict((val, i) for i, val in enumerate(unique_vals))
            map_list.append(mapped_vals)
            processed_df[attr].replace(mapped_vals, inplace=True)

    processed_df.fillna(0, inplace=True)

    for col in processed_df.columns[:-1]:
        max_v = df[col].astype('float').max()
        min_v = df[col].astype('float').min()
        range_v = max_v - min_v
        processed_df[col] = (processed_df[col] - min_v) / range_v

    return processed_df, map_list


def create_attr_dict(schema):
    '''
    Creates a dictionary mapping attribute names to their information (type, choices, attribute
    # entropy and test values-for continuous attributes)
    '''
    attr_dict = {}

    for attr in schema:
        if attr.type == Feature.Type.BINARY or attr.type == Feature.Type.CLASS:
            attr_dict[attr.name] = {
                type_key: attr.type, choices_key: (True, False)}

        elif attr.type == Feature.Type.NOMINAL:
            attr_dict[attr.name] = {
                type_key: attr.type, choices_key: attr.values}

        elif attr.type == Feature.Type.CONTINUOUS:
            attr_dict[attr.name] = {type_key: attr.type}

    return attr_dict


def attr_type(attr_dict, attr_name):
    return attr_dict[attr_name][type_key]


def attr_choices(attr_dict, attr_name):
    return attr_dict[attr_name][choices_key]


def create_folds(df, random_seed=12345):
    """
    5-fold stratified cross validation:
    Since we want an equal number of true and false examples in each data set,
    start by creating two dataframes to separate true and false class labels.
    Shuffle these dataframes and then split each into 5 smaller subarrays.
    Then, take true[i] and concatenate with false[i] to create folds
    """
    np.random.seed(random_seed)

    folds = []
    false = df[df[CLASS_NAME] == False]
    true = df[df[CLASS_NAME] == True]

    # shuffle the indexes for a random selection
    f_idxs = np.array(range(false.shape[0]))
    t_idxs = np.array(range(true.shape[0]))
    np.random.shuffle(f_idxs)
    np.random.shuffle(t_idxs)

    # chunk examples into 5 equal length subarrays
    false_div = np.array_split(f_idxs, 5)
    true_div = np.array_split(t_idxs, 5)

    # concat true and false subarrays to make folds
    for i in range(5):
        f_append = false.iloc[false_div[i]]
        t_append = true.iloc[true_div[i]]
        folds.append(pd.concat([f_append, t_append]))

    return folds


def train_test_split(folds, test_i):
    '''
    Given the five folds, this algorithm concatenates 4 of the folds together
    for training and uses the other for testing
    '''
    folds_cpy = deepcopy(folds)
    test = folds_cpy.pop(test_i)
    train = pd.concat(folds_cpy)

    return train, test
