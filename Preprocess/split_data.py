from read_file import *
from filter_function import *
from pprint import pprint
import random 
random.seed(1)

class SplitData:
    def __init__(self, data: list) -> None:
        try:
            data.remove([])
        except:
            pass
        self.data = data
        self.size = len(data)

    def splitting(self, train_percent: float):
        train = []
        train_size = int(train_percent * self.size)
        data_copy = list(self.data)
        for i in range(train_size):
            random_index = random.randrange(len(data_copy))
            train.append(data_copy[random_index])
            data_copy.pop(random_index)
        test = data_copy
        return train, test

    def createKFoldIndex(self, data_size, k_fold):
        all_index = [i for i in range(data_size)]
        all_index_copy = list(all_index)
        fold_size = int( data_size / k_fold)
        output_train, output_test = [], []
        for _ in range(k_fold):
            output_test.append(random.sample(all_index, fold_size))
            output_train.append([ele for ele in all_index_copy if ele not in output_test[_]])
        return output_train, output_test

    def crossValidation(self, k_fold):
        train_folds, test_folds = self.createKFoldIndex(self.size, k_fold)
        fold_size = int( self.size / k_fold)
        data_copy = list(self.data)
        trains, tests = [], []
        for fold in range(k_fold):
            train, test = [], []
            for i, index in enumerate(train_folds[fold]):
                train.append(data_copy[index])
            for i, index in enumerate(test_folds[fold]):
                test.append(data_copy[index])
            trains.append(train)
            tests.append(test)
        return trains, tests
    
if __name__ == "__main__":
    csv_dir = r"iris.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)
    data = removeEmptyRecord(data)

    split = SplitData(data)

    #Split by train/test
    train, test = split.splitting(0.5)
    pprint(train[:10])
    pprint(test[:10])

    print("#"*20)

    #Cross Validation split
    trains, tests = split.crossValidation(5)
    pprint(trains[0][:10])
    pprint(tests[0][:10])