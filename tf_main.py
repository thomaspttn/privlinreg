from utils import data_to_dataframe, create_attr_dict
from utils import process_data, create_folds, train_test_split
from privlinreg import PrivLinReg
from perf_tester import metrics_for_fold, auc
from mldata import parse_c45
import os
import sys
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# CONFIGURE HYPERPARAMETERS
np.random.seed(12345)
tf.random.set_seed(12345)

# Get command line argument for data location
argument_list = sys.argv[1:]
path = str(argument_list[0])
filename = os.path.basename(path)
filedir = path.replace(filename, '')
data = parse_c45(filename, filedir)

# Define epsilon value and type of noise
epsilon = float(argument_list[1])
noise_type = argument_list[2]

# Convert c45 data to DataFrame and create folds
unprocessed_df = data_to_dataframe(data)
attr_dict = create_attr_dict(data.schema)
df_whole, _ = process_data(unprocessed_df, attr_dict)
folds = create_folds(df_whole)

# Create a DataFrame to store important metrics
metrics_df = pd.DataFrame(columns=['fold', 'accuracy', 'precision', 'recall'])
metrics = []

# Perform 5-fold cross-validation
for i in range(len(folds)):
    print('Predicting Fold : ', i+1)
    train, test = train_test_split(folds, i)
    plr = PrivLinReg(train, epsilon=epsilon, noise_type=noise_type)
    trained_model = plr.train_model(epochs=500, learning_rate=0.001)
    acc, precision, recall, auc = metrics_for_fold(test, model=trained_model)
    metrics.append([acc, precision, recall, auc])

metrics_df = pd.DataFrame(np.array(metrics),
                          columns=['accuracy', 'precision', 'recall', 'auc'])

mean_vals = metrics_df.mean(axis=0).to_numpy()
std_vals = metrics_df.std(axis=0).to_numpy()

# Display results
print('')
print(metrics_df)
print('')

print('Mean Val +/- STD')
print('accuracy :: %0.4f +/- %0.4f' % (mean_vals[0], std_vals[0]))
print('precision :: %0.4f +/- %0.4f' % (mean_vals[1], std_vals[1]))
print('recall :: %0.4f +/- %0.4f' % (mean_vals[2], std_vals[2]))
print('auc :: %0.4f +/- %0.4f' % (mean_vals[3], std_vals[3]))
