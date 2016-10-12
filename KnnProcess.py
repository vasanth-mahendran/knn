import math
import pandas as pandas
from Kfcv import Kfcv
from collections import OrderedDict
from itertools import islice
import time


k = 1
kfcv_k = 10
distance_metric = 'polynomial'
data_set = 'glass'

class KnnProcess(object):
    def __init__(self):
        start_time = time.time()
        # Un comment this manually give parameters
        #self.get_inputs()
        print("--- Starting: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        kfcv_obj = Kfcv(data_set, kfcv_k)
        print("--- Parsed Files: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        if kfcv_obj.get_partitions() is not None:
            validations = pandas.DataFrame([], columns=list(kfcv_obj.get_records().columns.values))
            partitions = list(kfcv_obj.get_partitions())
            for idx, validation in enumerate(partitions):
                training_list = list(partitions)
                del training_list[idx]
                self.training = KnnProcess.flatten_list_of_lists(training_list)
                knn_validate = self.do_knn(validation)
                validation = validation.merge(knn_validate, left_index=True, right_index=True)
                validations = validations.append(validation)
            accuracy = KnnProcess.calculate_accuracy(validations['class'], validations['predicted_class'])
            print('Accuracy of KNN for data set %s is %s with k %s and distance metric %s , time taken is %s' %(data_set, accuracy*100,k,distance_metric,round(((time.time() - start_time) / 60), 2)))
        else:
            print('Enter proper data set name')
        print("--- Knn Done: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    def get_inputs(self):
        global k
        global distance_metric
        global data_set
        global kfcv_k

        try:
            print("Distance Metric\n1.euclidean\n2.polynomial kernel\n3.radial basis kernel")
            distance_metric_sucess = False
            data_set_sucess = False
            k_sucess = False
            kfcv_k_sucess = False
            while True:
                distance_metric_input = int(input('\nEnter choice 1 or 2 or 3:'))
                if distance_metric_input == 1:
                    distance_metric = 'euclidean'
                    distance_metric_sucess = True
                elif distance_metric_input == 2:
                    distance_metric = 'polynomial'
                    distance_metric_sucess = True
                elif distance_metric_input == 3:
                    distance_metric = 'radial'
                    distance_metric_sucess = True
                else:
                    print('Invalid option for distance metric')
                if distance_metric_sucess:
                    break
            while True:
                k = int(input('\nEnter input parameter k for knn:'))
                if k>0:
                    k_sucess=True
                else:
                    print('Invalid option k for knn (select k greater than 0)')
                if k_sucess:
                    break
            while True:
                kfcv_k = int(input('\nEnter input parameter k for kfcv:'))
                if kfcv_k>0:
                    kfcv_k_sucess=True
                else:
                    print('Invalid option k for kfcv (select k greater than 0)')
                if kfcv_k_sucess:
                    break
            print("\nData set\n1.ecoli\n2.yeast\n3.glass")
            while True:
                data_set_input = int(input('\nEnter choice 1 or 2 or 3:'))
                if data_set_input == 1:
                    data_set = 'ecoli'
                    data_set_sucess = True
                elif data_set_input == 2:
                    data_set = 'yeast'
                    data_set_sucess = True
                elif data_set_input == 3:
                    data_set = 'glass'
                    data_set_sucess= True
                else:
                    print('Invalid option for Data set')
                if data_set_sucess:
                    break
        except:
            print("problem while processing the files")
            raise Exception('Please enter proper inputs')

    def do_knn(self, validation):
        predicted = validation.apply(self.predict_class, axis=1)
        return predicted

    def predict_class(self, validate_row):
        distance = dict()
        training = self.training
        for training_index, training_row in training.iterrows():    
            distance[training_index] = KnnProcess.calculate_distance(distance_metric, list(training.columns.values),
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
        items = sliced_distance_ordered.items()
        for idx, distance in items:
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
        elif d_metric == 'polynomial':
            return KnnProcess.polynomial_kernel(columns, validate_row, training_row)

    @staticmethod
    def euclidean_distance(columns, validate_row, training_row):
        euclidean_sum = 0
        if 'class' in columns:
            columns.remove('class')
        if 'Sequence' in columns:
            columns.remove('Sequence')
        if 'Id' in columns:
            columns.remove('Id')
        for column in columns:
            euclidean_sum += math.pow(validate_row[column] - training_row[column], 2)
        return math.sqrt(euclidean_sum)
    
    @staticmethod
    def polynomial_kernel(columns, validate_row, training_row):
        dot_product = 40
        if 'class' in columns:
            columns.remove('class')
        if 'Sequence' in columns:
            columns.remove('Sequence')
        if 'Id' in columns:
            columns.remove('Id')

        for column in columns:
            dot_product += validate_row[column] * training_row[column]
        return math.pow((1 + dot_product),len(columns))
    
    @staticmethod
    def run_batch():
        global k
        global data_set
        global distance_metric
        distance_metrics_list = ['euclidean','polynomial']
        k_list = [1,2,3,5,10,15]
        data_set_list = ['ecoli','yeast','glass']

        for data_set_batch in data_set_list:
            for distance_metric_batch in distance_metrics_list:
                for k_batch in k_list:
                    k=k_batch
                    data_set = data_set_batch
                    distance_metric = distance_metric_batch
                    knn_process_obj = KnnProcess()

# Un comment this give manual parameters   
#knn_process_obj = KnnProcess()
# Un comment this  runs the knn for differernt measures with different k and data set
batch_start_time = time.time()
print("--- Starting batch: %s minutes ---" % round(((time.time() - batch_start_time) / 60), 2))
KnnProcess.run_batch()
print("--- Ending batch: %s minutes ---" % round(((time.time() - batch_start_time) / 60), 2))