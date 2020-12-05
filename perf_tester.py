import numpy as np
import pandas as pd
from utils import CLASS_NAME

RANDOM_SEED = 12345
CONFIDENCE_LBL = 'CONFIDENCE'
EPOCHS = 10000
LR = 0.00005


def metrics_for_fold(testing_df, model):
    '''
    Determine the relevant metrics for one fold of the validation

    :param DataFrame testing_df: The dataframe to test on
    :param keras.Model model: The Keras model to use for prediction
    :return: accuracy, precision, recall, auc metrics
    '''
    true_classes = []
    conf_levels = []
    num_examples = testing_df.shape[0]
    total_pos = testing_df[testing_df[CLASS_NAME].astype('int') == 1].shape[0]
    tp_plus_tn = 0
    true_pos = 0
    false_pos = 0

    for row in range(num_examples):
        true_cl = testing_df.iloc[row][CLASS_NAME]
        example = testing_df.iloc[row, :-1].to_numpy(dtype='float32')
        confidence = model.predict(example.reshape(1, testing_df.shape[1]-1),
                                   verbose=0)
        predicted_cl = 1 if confidence >= 0.5 else 0

        true_classes.append(true_cl)
        conf_levels.append(confidence)

        if true_cl == predicted_cl:
            tp_plus_tn += 1

        if true_cl and predicted_cl:
            true_pos += 1
        elif (true_cl == False) and (predicted_cl == True):
            false_pos += 1

    accuracy = np.round(tp_plus_tn / num_examples, 3)
    precision = 0.0 if (
        true_pos + false_pos) == 0 else np.round(true_pos / (true_pos + false_pos), 3)
    recall = 0.0 if total_pos == 0 else np.round(true_pos / total_pos, 3)

    auc_v = auc(true_classes, conf_levels, num_examples)

    return accuracy, precision, recall, auc_v


def auc(true_classes, conf_levels, num_examples):
    '''
    Determine the AUC metric

    :return: AUC
    '''
    roc_df = pd.DataFrame()
    roc_df[CLASS_NAME] = true_classes
    roc_df[CONFIDENCE_LBL] = conf_levels
    roc_df.sort_values(CONFIDENCE_LBL, ascending=False,
                       inplace=True, ignore_index=True)

    total_pos = roc_df[roc_df[CLASS_NAME].astype('int') == 1].shape[0]
    total_neg = num_examples - total_pos
    false_pos = 0
    true_pos = 0

    prev_fp = 0
    prev_tp = 0
    total_area = 0

    for i in range(num_examples):
        if roc_df[CLASS_NAME][i]:
            true_pos += 1
        else:
            false_pos += 1

        cur_fp = false_pos / total_neg
        cur_tp = true_pos / total_pos
        total_area += trapezoid_area(a=prev_tp, b=cur_tp, h=(cur_fp - prev_fp))
        prev_fp = cur_fp
        prev_tp = cur_tp

    return np.round(total_area, 3)


def trapezoid_area(a=0, b=0, h=0):
    return ((a + b) * h) / 2
