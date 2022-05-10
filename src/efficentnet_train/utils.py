import numpy as np
import json
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def save_dict_to_json(path_file,data_dict):
  with open(path_file, 'w') as fp:
    json.dump(data_dict, fp)