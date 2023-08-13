from read_file import *
from filter_function import *
from pprint import pprint
import random 
random.seed(1)

class SplitDataFeatureTarget:
    def __init__(self, data: list) -> None:
        self.data = data
        self.size = len(data)
    
    def splitting(self, train_percent: float):
        X_train, Y_train = [], []
        train_size = int(train_percent * self.size)
        data_copy = list(self.data)
        for i in range(train_size):
            random_index = random.randrange(len(data_copy))
            X_train.append(data_copy[random_index][:-1])
            Y_train.append(data_copy[random_index][-1])
            data_copy.pop(random_index)
        
        X_test = [row[:-1] for row in data_copy if row != []]
        Y_test = [row[-1] for row in data_copy if row != []]
        return X_train, Y_train, X_test, Y_test

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
        Xs_train, Ys_train, Xs_test, Ys_test = [], [], [], []
        for fold in range(k_fold):
            X_train, Y_train, X_test, Y_test = [], [], [], []
            for i, index in enumerate(train_folds[fold]):
                X_train.append(data_copy[index][:-1])
                Y_train.append(data_copy[index][-1])
            for i, index in enumerate(test_folds[fold]):
                X_test.append(data_copy[index][:-1])
                Y_test.append(data_copy[index][-1])
            Xs_train.append(X_train)
            Ys_train.append(Y_train)
            Xs_test.append(X_test)
            Ys_test.append(Y_test)
        return Xs_train, Ys_train, Xs_test, Ys_test

if __name__ == "__main__":
    csv_dir = r"iris.csv"
    data = loadDataFromCSV(csv_dir)
    data = convertToFloat(data)
    data = removeEmptyRecord(data)

    split = SplitDataFeatureTarget(data)

    #Split by train/test
    X_train, Y_train, X_test, Y_test = split.splitting(0.5)
    print(X_train[:10])
    print(Y_train[:10])

    print("#"*20)

    #Cross Validation split
    Xs_train, Ys_train, Xs_test, Ys_test = split.crossValidation(5)
    print(Xs_train[0][:10])
    print(Ys_train[0][:10])
