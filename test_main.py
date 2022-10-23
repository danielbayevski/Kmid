from unittest import TestCase
import numpy
from Kmid.main import load_data, predictFile
import os


class Test(TestCase):
    def test_predict_file (self):
        orig_files = os.scandir("test_sheets/original")
        test_files = "test_sheets/altered"
        new_directory = "test_sheets/predicted"
        for file in orig_files:
            col_num = load_data(file.path, endRows = 1).shape[1]
            data = load_data(file.path, cols = [col_num - 1])

            unique_first_iteration_pairs = {}
            for i, name in enumerate(data[data.columns[0]].unique()):
                unique_first_iteration_pairs.update({name: i})

            k = data.nunique()[0]
            predictFile(test_files + "/" + file.name, k, 0,
                        new_directory)
            col_num = load_data(new_directory + "/" + file.name,
                                endRows = 1).columns.size
            predictions = load_data(new_directory + "/" + file.name,
                                    cols = [col_num - 1])
            prediction_array = numpy.zeros((k, k), dtype = int)

            for i, pred in enumerate(predictions["grouping"]):
                name_iterator = unique_first_iteration_pairs[data.loc[i][0]]
                prediction_array[name_iterator][int(pred)] += 1

            pred_accuracy = 0

            for i in range(k):
                pred_accuracy += prediction_array[i].max() / \
                                 prediction_array[i].sum()

            pred_accuracy /= k
            print(file.name + " accuracy: " + str(pred_accuracy))


