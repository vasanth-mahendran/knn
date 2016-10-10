import math
import pandas as pandas
from Kfcv import Kfcv
from collections import OrderedDict
from itertools import islice
import time

start_time = time.time()
k = 10
kfcv_k = 10
distance_metric = 'euclidean'
data_set = 'ecoli'


class KnnProcess(object):
    def __init__(self):
        self.get_inputs()
        print("--- Starting: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        kfcv_obj = Kfcv(data_set, kfcv_k)
        print("--- Parsed Files: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        if kfcv_obj.get_partitions() is not None:
            validations = []
            partitions = list(kfcv_obj.get_partitions())
            i = 0
            for idx, validation in enumerate(partitions):
                training = list(partitions)
                del training[idx]
                self.training = KnnProcess.flatten_list_of_lists(training)
                validation = validation.merge(self.do_knn(validation), left_index=True, right_index=True)
                accuracy = KnnProcess.calculate_accuracy(validation['class'], validation['predicted_class'])
                validations.append(validation)
                i += 1
                print('Accuracy for partition ', i, ' is ', accuracy)
            print(KnnProcess.flatten_list_of_lists(validations))
        else:
            print('Enter proper data set name')
            print("--- Knn Done: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    def get_inputs(self):
        global k
        global distance_metric
        global data_set
        global kfcv_k

        distance_metric = input('Enter input for distance_metric:')
        k = int(input('Enter input parameter k for knn:'))
        kfcv_k = int(input('Enter input parameter k for kfcv:'))
        data_set = input('Enter input for data_set:')

    def do_knn(self, validation):
        validation = pandas.DataFrame(validation)
        predicted = validation.apply(self.predict_class, axis=1)
        return predicted

    def predict_class(self, validate_row):
        distance = dict()
        training = self.training
        if distance_metric == 'euclidean':
            for training_index, training_row in training.iterrows():
                distance[training_index] = KnnProcess.calculate_distance(distance_metric, training.columns,
                                                                         validate_row, training_row);
            distance_ordered = OrderedDict(sorted(distance.items(), key=lambda t: t[1]))
            predicted_class = KnnProcess.get_class(distance_ordered, training)
            return pandas.Series(dict(predicted_class=predicted_class))

    @staticmethod
    def calculate_accuracy(labeled_class, predicted_class):
        correct = 0
        for idx, row in labeled_class.iteritems():
            if row == predicted_class[idx]:
                correct += 1
        return correct / len(labeled_class)

    @staticmethod
    def flatten_list_of_lists(lst):
        df = pandas.concat(lst, ignore_index=True)
        return df

    @staticmethod
    def get_class(distance_ordered, training):
        sliced_distance = islice(distance_ordered.items(), k)
        sliced_distance_ordered = OrderedDict(sliced_distance)
        label_classes = dict()
        for idx, distance in sliced_distance_ordered.items():
            label_class = training.iloc[idx]['class']
            if label_class in label_classes.keys():
                label_classes[label_class] += 1
            else:
                label_classes[label_class] = 1
        return max(label_classes, key=label_classes.get)

    @staticmethod
    def calculate_distance(d_metric, columns, validate_row, training_row):
        if d_metric == 'euclidean':
            return KnnProcess.euclidean_distance(columns, validate_row, training_row)

    @staticmethod
    def euclidean_distance(columns, validate_row, training_row):
        euclidean_sum = 0
        for column in columns:
            if column != 'class' and column != 'Sequence' and column != 'Id' and column != 'predicted_class':
                euclidean_sum += math.pow((float(validate_row[column]) - float(training_row[column])), 2)
        return math.sqrt(euclidean_sum)


knn_process_obj = KnnProcess()
