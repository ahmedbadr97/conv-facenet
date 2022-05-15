import sys
import time

from .utils import euclidean_distance
import pandas as pd
import os


def model_test(features_vectors_dict, dataset_frame, threshold=0.5, results_path=None):
    error = 0
    data_cnt = dataset_frame.count()[0]
    cnt = 0.0
    # row --> predicted Class  , cols --> true Class
    confusion_matrix = [[0, 0], [0, 0]]
    false_positive_rows = []
    false_negative_rows = []
    total_dist_same_persons = 0.0
    total_dist_diff_persons = 0.0
    avg_dist_same_persons = 0.0
    avg_dist_diff_persons = 0.0

    time_sum = 0.0
    for _, row in dataset_frame.iterrows():
        ts = time.time()

        img1Path = row[0]
        img2Path = row[1]

        emp1 = features_vectors_dict[img1Path]
        emp2 = features_vectors_dict[img2Path]

        result = euclidean_distance(emp1, emp2)
        pred_dist = 0
        actual_dist = row[2]
        if result > threshold:
            pred_dist = 1

        if pred_dist != int(row[2]):
            error += 1
        confusion_matrix[pred_dist][actual_dist] += 1
        if pred_dist == 0 and actual_dist == 1:
            false_positive_rows.append(row)
        if pred_dist == 1 and actual_dist == 0:
            false_negative_rows.append(row)
        if actual_dist == 0:
            avg_dist_same_persons += result
            total_dist_same_persons += 1
        if actual_dist == 1:
            avg_dist_diff_persons += result
            total_dist_diff_persons += 1

        cnt += 1.0
        finished = int((cnt * 10) / data_cnt)

        remaining = 10 - finished
        te = time.time()
        time_sum += (te - ts)
        avg_time = time_sum / cnt
        time_remaing = avg_time * (data_cnt - cnt)
        accuracy = ((cnt - error) * 1.0 / cnt) * 100.0
        sys.stdout.write("\r Testing  [" + str(
            "=" * finished + str("." * remaining) + "] time remaining = " + str(
                time_remaing / 60.0)[:8]) + " Accuracy =" + str(round(accuracy, 3)))

    avg_dist_same_persons /= total_dist_same_persons
    avg_dist_diff_persons /= total_dist_diff_persons

    accuracy = ((cnt - error) * 1.0 / cnt) * 100.0
    print("Accuracy now equal --> {:.4f}%".format(accuracy))

    confusion_table = pd.DataFrame(confusion_matrix, index=['Predicted True', 'Predicted False'],
                                   columns=['Actual True', 'Actual False'])
    precision = (confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + 1)
    recall = (confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0] + 1)
    beta = 0.5
    beta_squared = beta ** 2
    fbeta_score = ((1 + beta_squared) * (precision * recall)) / ((beta_squared * precision) + recall + 1)
    error_matrix = [['processed rows', cnt],
                    ['Model accuracy on Proceed Faces %', round(accuracy, 3)],
                    ['False Positive', confusion_matrix[0][1]],
                    ['False Negative', confusion_matrix[1][0]],
                    ['precision', precision],
                    ['recall', recall],
                    ['fbeta-score', fbeta_score],
                    ['avg same person distance', avg_dist_same_persons],
                    ['avg diff person distance', avg_dist_diff_persons],
                    ['Model tolerance', threshold]
                    ]
    model_name = "efficentnet"

    error_table = pd.DataFrame(error_matrix, columns=['Mertic', 'Value'])

    if results_path is not None:
        file_name = f"{model_name}__err.csv"
        full_path = os.path.join(results_path, file_name)
        error_table.to_csv(full_path)

        file_name = f"{model_name}_conv.csv"
        full_path = os.path.join(results_path, file_name)
        confusion_table.to_csv(full_path)
    return error_table, confusion_table
