from .utils import findCosineDistance
import pandas as pd
import os
from os.path import join as join_pth
from PIL import Image
import numpy as np
import torch



def model_evaluation(dataVectorsDict, datasetFrame, threshold=0.5):
    error = 0
    data_cnt = datasetFrame.count()[0]
    cnt = 0.0
    # row --> predicted Class  , cols --> true Class
    confusion_matrix = [[0, 0], [0, 0]]
    cycle_cnt = 0
    err_rows = []
    false_postive_rows = []
    false_negative_rows = []
    total_dist_same_persons = 0.0
    total_dist_diff_persons = 0.0
    avg_dist_same_persons = 0.0
    avg_dist_diff_persons = 0.0

    for _, row in datasetFrame.iterrows():
        # img1Path=data_set_path+'/'+row[0]
        img1Path = row[0]
        img2Path = row[1]

        try:
            emp1 = dataVectorsDict[img1Path]
            emp2 = dataVectorsDict[img2Path]
        except:
            emp1 = emp2 = None
        if emp1 == None or emp2 == None:
            err_rows.append(row)
            continue
        result = findCosineDistance(emp1, emp2)
        pred_dist = 0
        actual_dist = row[2]
        if result > threshold:
            pred_dist = 1

        if pred_dist != int(row[2]):
            error += 1
        confusion_matrix[pred_dist][actual_dist] += 1
        if pred_dist == 0 and actual_dist == 1:
            false_postive_rows.append(row)
        if pred_dist == 1 and actual_dist == 0:
            false_negative_rows.append(row)
        if actual_dist == 0:
            avg_dist_same_persons += result
        if actual_dist == 1:
            avg_dist_diff_persons += result

        cnt += 1.0
        cycle_cnt += 1
        if cycle_cnt == 10000:
            print("data proceesed -> {:.2f}%".format((cnt / data_cnt) * 100.0))
            accuracy = ((cnt - error) * 1.0 / cnt) * 100.0
            print("Accuracy now equal --> {:.4f}%".format(accuracy))
            cycle_cnt = 0

    avg_dist_same_persons /= (cnt / 2)
    avg_dist_diff_persons /= (cnt / 2)

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
                    ['No unProceesed Faces', len(err_rows)],
                    ['Model accuracy on Procced Faces %', round(accuracy, 3)],
                    ['False Postive', confusion_matrix[0][1]],

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

    folder_pth = ("\\models_evaluation_data\\face_recognition")
    if not os.path.exists(folder_pth):
        os.mkdir(folder_pth)
    # TODO add dateTime to the filename
    file_name = f"{model_name}__err.csv"
    full_path = os.path.join(folder_pth, file_name)
    error_table.to_csv(full_path)

    file_name = f"{model_name}_conv.csv"
    full_path = os.path.join(folder_pth, file_name)
    confusion_table.to_csv(full_path)
    return error_matrix, confusion_matrix
